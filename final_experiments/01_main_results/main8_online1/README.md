# main8_online1

Single-thread staging for the main eight-dataset experiment. The launcher is
`../run_main8_online1.py`; it reuses the 24-thread launcher and changes only
the online/search thread count to 1.

Run both backends:

```bash
./run_all.sh
```

Run cells separately:

```bash
./run_hnswlib.sh
./run_faiss.sh
```
