# Pile-Dataset
### Requirements
I recommend using a good CPU, but besides that you **will** need:
- At least **1000 Gib**, optimally **2000 Gib** of free storage
- At least **32 Gib** of RAM
- A very good internet connection 

### Download
The Pile dataset is an **825 Gib** (around **420 Gib** (nice) compressed) open source language model dataset.
You can download it from [here](https://pile.eleuther.ai/)

**Please download to ``data/*.zst``**

I recommend using ``wget https://the-eye.eu/public/AI/pile/train/<num 00-29>.jsonl.zst &``
to download in a sub-process.
Each file should be around **14 Gib** big.

### Decompressing
They are compressed using the zstandard.
To decompress use ``python decompress.py``, this may take a bit.
The actual size of each file should be around **42 Gib**.

**NOTE** : ```decompress.py``` uses multiprocessing to decompress all at the same time.
Use ``decompress_alt.py`` to decompress without multiprocessing.

After decompressing there should be ``.json`` files in addition to the ``.zst`` files. 
You may delete those.