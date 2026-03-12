# Object Dataset Pipeline Scripts

## 1) Install dependencies

```bash
python3 -m pip install -r requirements-objects.txt
```

## 2) Fetch datasets

```bash
bash scripts/fetch_ycb.sh data/objects/ycb/raw
bash scripts/fetch_hope.sh data/objects/hope/raw
```

## 3) Build inventories

```bash
python3 scripts/scan_object_inventory.py
```

## 4) Build manifest

```bash
python3 scripts/build_object_manifest.py
```

## 5) Validate meshes

```bash
python3 scripts/validate_meshes.py
```
