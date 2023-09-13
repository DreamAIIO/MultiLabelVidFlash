from imports import*

def setify(o): return o if isinstance(o,set) else set(list(o))

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    if folders is None:
        folders = list([])
    path = Path(path)
    if extensions is not None:
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return list(res)

def oneHot(vids):
    l = []
    for v in vids:
        n = v.parent.name
        if n == "archery":
            lab = [1,0,0,0,0,1,0]
        elif n == "bowling":
            lab = [0,1,0,0,0,1,0]
        elif n == "flying_kite":
            lab = [0,0,1,0,0,0,1]
        elif n == "high_jump":
            lab = [0,0,0,1,0,0,1]
        elif n == "marching":
            lab = [0,0,0,0,1,0,1]
        l.append(lab)
    l = np.array(l)
    l = torch.from_numpy(l)
    return l

def createMultiLabelDf (data_path):
    data_path = Path(data_path)
    vids = get_files(data_path, extensions=['.mp4'])
    l = oneHot(vids)
    vids = [str(vid).replace(str(data_path)+"/","") for vid in vids]
    
    df = pd.DataFrame({'video':vids, 'archery':l[:,0], 'bowling':l[:,1],
                   'flying_kite':l[:,2], 'high_jump':l[:,3],
                   'marching':l[:,4], 'indoor':l[:,5],
                   'outdoor':l[:,6]})
    
    df = df.sample(frac=1)
    
    df.to_csv(str(data_path)+"/"+str(data_path.name)+".csv", index=False)
    
