from glob import glob
import zstandard as zst


def decompress(filename):
    output_path = filename + '.json'
    print('Decompressing', filename, 'to', output_path)
    dctx = zst.ZstdDecompressor()
    with open(filename, 'rb') as ifh, open(output_path, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)
    print('Decompressed', filename, 'to', output_path)


for file in glob('data/*.zst'):
    decompress(file)
