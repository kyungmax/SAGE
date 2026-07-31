# SAGE prebuilt indexes

Place prebuilt HNSW indexes for the artifact in this directory.

Expected default layout:

```text
index/
  faiss_m32_efc500_main8_20260707/
    darth/
      index/
        <dataset-stem>/
          <dataset-stem>.M32.efC500.index
  <hnswlib index files>
```

The root README contains the download instructions. Large index binaries should
stay out of git; `.gitignore` keeps everything under this directory ignored
except this file.
