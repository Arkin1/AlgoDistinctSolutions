import multiprocessing
from tqdm import tqdm
from typing import Any, Union

def _func_with_idx(el_with_func_idx):
    idx, el, func = el_with_func_idx

    return (idx, func(el))


def tqdm_multiprocess_map(func, elements:list[Any], max_workers:int, chunksize:int):

    elements_with_idx = [(idx, el, func) for idx, el in enumerate(elements)]

    with multiprocessing.Pool(max_workers) as pool:
        processed_with_idx = list(tqdm(pool.imap_unordered(_func_with_idx, elements_with_idx, chunksize = chunksize), total= len(elements_with_idx)))

    processed_with_idx = sorted(processed_with_idx, key = lambda x: x[0])

    return [p for _, p in processed_with_idx]
