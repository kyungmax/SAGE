#!/usr/bin/env python3
"""Check the main-six workspace and inputs without loading or searching a graph."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import struct
import sys

import h5py
from main6_common import RUN_ROOT, effective_config, resolve_datasets, validate_policy_config

ROOT=Path(__file__).resolve().parent
MAIN=RUN_ROOT
CONFIG=ROOT/'config/main6_online24'
PREPARATION=RUN_ROOT/'preparation/main6_online24'
SAGE_ROOT=next(p for p in ROOT.parents if (p/'sage_env.sh').is_file() and (p/'experiments_scripts').is_dir())


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def identity(p):
    path=Path(p).resolve(strict=True);s=path.stat()
    if s.st_size==0:raise ValueError(f'Empty input: {path}')
    return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def inspect(ds, required_k):
    data=identity(ds['dataset_path']);indexes={}
    with h5py.File(data['path'],'r') as f:
        shapes={name:list(f[name].shape) for name in ['train','test','neighbors']}
        assert shapes['train']==[ds['train_count'],ds['dimension']],ds['dataset']
        assert shapes['test']==[ds['query_count'],ds['dimension']],ds['dataset']
        assert shapes['neighbors'][0]==ds['query_count'] and shapes['neighbors'][1]>=required_k
        external=[]
        for name,offset,size in f['train'].external or []:
            p=Path(name.decode() if isinstance(name,bytes) else name)
            if not p.is_absolute():p=Path(data['path']).parent/p
            external.append(identity(p)|dict(offset=int(offset),declared_bytes=int(size)))
    for backend in ['faiss','hnswlib']:
        expected=ds[backend+'_index']
        if expected.get('bytes') is None:
            raise ValueError(f'{backend} graph awaits an approved build and input binding')
        actual=identity(expected['path'])
        assert actual['bytes']==expected['bytes'],(ds['dataset'],backend,'size changed')
        assert actual['mtime_ns']==expected['mtime_ns'],(ds['dataset'],backend,'mtime changed')
        indexes[backend]=actual
    with Path(indexes['hnswlib']['path']).open('rb') as f:header=f.read(struct.calcsize('<6QiI3QdQ'))
    values=struct.unpack('<6QiI3QdQ',header)
    _,capacity,count,stride,label_offset,data_offset,_,_,max_m,max_m0,m,_,efc=values
    assert count==ds['train_count'] and count<=capacity
    assert (label_offset-data_offset)//4==ds['dimension']
    assert m==ds['M'] and efc==ds['efConstruction']
    assert hashlib.sha256(header).hexdigest()==ds['hnswlib_header']['sha256']
    return dict(dataset=ds['dataset'],dataset_file=data,shapes=shapes,indexes=indexes,
        external_train_sources=external,index_fingerprints='file size/mtime plus hnswlib serialized header; no full graph scan')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PREPARATION/'preflight.json')
    args=parser.parse_args()
    errors=[];checks=[];binaries=[]
    config=read(CONFIG/'experiment.json');dataset_config=read(CONFIG/'datasets.json')
    validate_policy_config(config)
    assert config['k_values']==[10,100]
    assert config['exclude_datasets_by_k']=={'100':['youtube-15M-angular']}
    assert config['status'] in ('prepared_not_run','awaiting_final_approval','approved_not_run') and not config['benchmark_launched']
    assert len(config['datasets'])==len(set(config['datasets'])) and config['calibration']['offline_min_ef']==64
    assert config['offline_threads']==config['online_threads']==24
    assert config['timed_repeats']==3 and config['k']==10
    assert [d['dataset'] for d in dataset_config['datasets']]==config['datasets']
    for d in resolve_datasets(CONFIG/'datasets.json',config):
        required_k=max(k for k in config['k_values'] if d['dataset'] in effective_config(CONFIG/'experiment.json',k)['datasets'])
        try:checks.append(inspect(d,required_k))
        except Exception as exc:errors.append(dict(dataset=d['dataset'],error=repr(exc)))
    binary_names=[
        'faiss/build_sage_avx512/faiss/python/faiss/_swigfaiss_avx512.so',
        'baselines/pip/build/paper095/faiss/python/faiss/_swigfaiss_avx512.so',
        'baselines/abs/build/libpaper_abs.so']
    extensions=list((SAGE_ROOT/'hnswlib').glob('hnswlib*.so'))
    if len(extensions)!=1:errors.append(dict(error='Expected one local hnswlib extension'))
    else:binary_names.append(str(extensions[0].relative_to(SAGE_ROOT)))
    for name in binary_names:
        try:
            p=SAGE_ROOT/name;actual=sha(p)
            binaries.append(dict(path=str(p),sha256=actual))
        except Exception as exc:errors.append(dict(binary=name,error=repr(exc)))
    artifacts=[]
    for folder in ['k10','k100']:
        for p in (MAIN/folder).rglob('*'):
            if p.is_file() and p.name!='.gitkeep':artifacts.append(str(p.relative_to(MAIN)))
    source_names=['experiments_scripts/main/run_baselines_main6.py',
        'experiments_scripts/main/tune_baselines.py',
        'faiss/faiss/impl/HNSW.cpp','faiss/faiss/impl/VisitedTable.cpp','faiss/faiss/IndexHNSW.cpp',
        'hnswlib/hnswlib/hnswalg.h','hnswlib/python_bindings/bindings.cpp',
        'experiments_scripts/common/final_policy.py','experiments_scripts/faiss/faiss_sage_index.py',
        'experiments_scripts/main/run_search_main6.py','experiments_scripts/main/main6_common.py',
        'baselines/pip/faiss_paper095/faiss/impl/VisitedTable.cpp','baselines/abs/faiss/faiss/impl/VisitedTable.cpp']
    sources={name:sha(SAGE_ROOT/name) for name in source_names}
    p=SAGE_ROOT/'experiments_scripts/common/final_policy.py'
    tree=ast.parse(p.read_text())
    grid=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_grid_for_cap')
    pending=list(config['implementation_work_remaining'])
    connected={cell: worker is not None and (ROOT/worker).is_file() for cell,worker in config['cell_workers'].items()}
    result=dict(passed=not errors,inputs_ready=not errors,ready_to_run=config.get('execution_authorized',False) and not errors and not pending and all(connected.values()) and not artifacts,status=config['status'],
        check_mode='current_inputs_and_builds',
        execution_authorized=config.get('execution_authorized',False),pending_builds=dataset_config.get('pending_builds',[]),
        k_scopes={str(k): effective_config(CONFIG/'experiment.json',k)['datasets'] for k in config['k_values']},
        no_ann_run=True,graph_loaded=False,calibration_run=False,benchmark_launched=False,
        time=datetime.datetime.now().astimezone().isoformat(),python=sys.executable,
        checked_datasets=checks,binaries=binaries,sources_sha256=sources,
        config_sha256={p.name:sha(p) for p in CONFIG.glob('*.json')},
        output_directories_empty=not artifacts,hnswlib_current_grid_implementation=ast.get_source_segment(p.read_text(),grid),
        implementation_work_remaining=pending,
        workers_connected={cell: worker is not None and (ROOT/worker).is_file()
                           for cell,worker in config['cell_workers'].items()},errors=errors)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','inputs_ready','ready_to_run','no_ann_run','errors']},indent=2))
    print(f"Checked {len(checks)} datasets and {len(binaries)} binaries. Report: {args.output}")
    return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(main())
