from pathlib import Path
from typing import Union, List, Tuple


def find_files(search_list: Union[str, Path, List[str], List[Path], Tuple[str], Tuple[Path]], ext: Union[str, List[str], Tuple[str]], recursive: bool = False):
    if isinstance(ext, (list, tuple)):
        ext = [e if e.startswith(".") else "."+e for e in ext ]
    else:
        ext = [ext] if str(ext).startswith(".") else ["."+ext]
    if isinstance(search_list, (list, tuple)):
        p = [Path(i) for i in search_list]
    else:
        p = [Path(search_list)]
    paths = []
    for j in p:
        if not str(j).startswith("http"):
            path = Path(j).resolve()
            if not path.exists():
                continue
        else:
            path = j
        if path.is_dir():
            if recursive:
                [paths.extend(path.rglob("*"+e)) for e in ext]
            else:
                [paths.extend(path.glob("*"+e)) for e in ext]
        elif path.suffix in ext:
            paths.append(path)
    return paths