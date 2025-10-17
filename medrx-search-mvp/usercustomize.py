import os, importlib, inspect, re, numbers

print(f"[INFO] usercustomize loaded from: {__file__}")

WRRF_ALPHA = float(os.environ.get("WRRF_ALPHA", os.environ.get("ALPHA", "60")))
W_BM25     = float(os.environ.get("WRRF_W_BM25", "1"))
W_DENSE    = float(os.environ.get("WRRF_W_DENSE", "1"))

def _log(msg):  print(f"[INFO] {msg}")
def _warn(msg): print(f"[WARN] {msg}")

def _looks_like_hits(x):
    # список кортежів (doc_id, score)
    if not isinstance(x, list) or not x:
        return False
    a = x[0]
    if not (isinstance(a, (tuple, list)) and len(a)==2):
        return False
    doc, sc = a
    return isinstance(sc, numbers.Number)

def _pick_two_hitlists(args_dict):
    """Повертає (hits1, hits2, top_k) або (None, None, None) якщо не зійшлось."""
    vals = list(args_dict.items())
    # вибрати дві перші, що "схожі" на список хітів
    hits = [(k,v) for k,v in vals if _looks_like_hits(v)]
    if len(hits) < 2:
        return None, None, None
    (k1,h1),(k2,h2) = hits[0], hits[1]
    # знайти top_k
    top_k = None
    for cand in ("top_k","top","k","limit"):
        if cand in args_dict and isinstance(args_dict[cand], numbers.Integral):
            top_k = int(args_dict[cand]); break
    if top_k is None:
        # fallback: розмір об’єднання
        top_k = len(set([d for d,_ in h1] + [d for d,_ in h2]))
    return h1, h2, top_k

def _weighted_rrf(bm25_hits, dense_hits, top_k, alpha, w_b, w_d):
    rb = {doc_id: i+1 for i, (doc_id, _) in enumerate(bm25_hits)}
    rd = {doc_id: i+1 for i, (doc_id, _) in enumerate(dense_hits)}
    all_ids = set(rb) | set(rd)
    scores = {}
    for doc_id in all_ids:
        s = 0.0
        if doc_id in rb: s += w_b / (alpha + rb[doc_id])
        if doc_id in rd: s += w_d / (alpha + rd[doc_id])
        scores[doc_id] = s
    fused = sorted(all_ids, key=lambda d: scores[d], reverse=True)[:int(top_k)]
    return [(d, scores[d]) for d in fused]

def _make_generic_wrapper(fn):
    sig = inspect.signature(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        self = bound.arguments.get("self")
        # спроба авто-виявлення форматів
        h1, h2, top_k = _pick_two_hitlists(bound.arguments)
        if h1 is None:
            return fn(*args, **kwargs)  # не наша історія
        alpha = getattr(self, "rrf_alpha", WRRF_ALPHA) if self is not None else WRRF_ALPHA
        w_b   = getattr(self, "w_bm25",   W_BM25)     if self is not None else W_BM25
        w_d   = getattr(self, "w_dense",  W_DENSE)    if self is not None else W_DENSE
        res = _weighted_rrf(h1, h2, top_k, alpha, w_b, w_d)
        if self is not None and not getattr(self, "_wrrf_logged_once", False):
            _log(f"WRRF fuse engaged (alpha={alpha}, w_bm25={w_b}, w_dense={w_d}) via {self.__class__.__name__}.{fn.__name__}")
            setattr(self, "_wrrf_logged_once", True)
        return res
    wrapper.__name__ = fn.__name__
    return wrapper

def _patch_module(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        _log(f"usercustomize: imported {mod_name}")
    except Exception as e:
        _warn(f"cannot import {mod_name}: {e}")
        return

    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and hasattr(obj, "build_from_dataframe"):
            # поставимо гіперпараметри на інстанс при build...
            orig_build = getattr(obj, "build_from_dataframe")
            def make_wrapped(orig):
                def wrapped(self, *a, **kw):
                    self.rrf_alpha = WRRF_ALPHA
                    self.w_bm25    = W_BM25
                    self.w_dense   = W_DENSE
                    _log(f"WRRF activated on {self.__class__.__name__}: alpha={self.rrf_alpha}, w_bm25={self.w_bm25}, w_dense={self.w_dense}")
                    return orig(self, *a, **kw)
                return wrapped
            setattr(obj, "build_from_dataframe", make_wrapped(orig_build))

            # ...а тут знайдемо і патчимо ймовірні методи злиття
            patched = []
            for attr, fn in vars(obj).items():
                if callable(fn) and re.search(r"(rrf|fuse|merge|comb|blend|fusion|rank)", attr, re.I):
                    try:
                        wrap = _make_generic_wrapper(fn)
                        setattr(obj, attr, wrap)
                        patched.append(attr)
                    except Exception as e:
                        _warn(f"skip patch {obj.__name__}.{attr}: {e}")
            if patched:
                _log(f"WRRF methods patched on {obj.__name__}: {', '.join(sorted(set(patched)))}")
            else:
                _warn(f"WRRF: no suitable fuse method found on {obj.__name__} (name scan)")
            # більше нічого не патчимо — цього достатньо

_patch_module("search.integrated_medical_assistant")
_patch_module("eval.evaluate_bench")
