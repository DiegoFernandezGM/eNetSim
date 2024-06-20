


--8<-- "README.md"

## Folder layout

    flexible_atrchitecture/         # main code
    docs/                           # documentation folder

## Pre-requisites

1. **Python**
Version >3.1 

## Documentation Editing

For help editing the documentation visit [mkdocs.org](https://www.mkdocs.org). To generate the docs locally type in the parent directory: `mkdocs serve`
and point the browser to [127.0.0.1.8000](http://127.0.0.1:8000)

You will need to install the `python-markdown-math` extension for rendering equations and the `markdown-callouts` extension for correctly displaying the warning and note blocks. All requirements can be installed automatically using

```bash
$ pip install -r docs/requirements.txt
```

You may need to install

```bash
$ pip install pip-tools
```

if you add new markdown extensions, edit the `requirements.in`  file under `docs/`

```bash
$ pip-compile requirements.in
```

