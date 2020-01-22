import json

fnm = ['utils/ff_test.json', 'utils/ff_val.json', 'utils/ff_train.json']
out_pref = ['utils/ff_test.txt', 'utils/ff_val.txt', 'utils/ff_train.txt']
fake_dir = ['FF_Deepfakes', 'FF_Face2Face', 'FF_FaceSwap', 'FF_NeuralTextures']

for f in range(len(fnm)):
    
    with open(fnm[f]) as json_file:
        
        data = json.load(json_file)
        
        fout = open(out_pref[f], 'w')
        [fout.write('FF_orig/' + x[0] + '.npy,1\n') for x in data]
        [fout.write('FF_orig/' + x[1] + '.npy,1\n') for x in data]
        for fk_dir in fake_dir:
            [fout.write(fk_dir + '/' + '_'.join(x)+'.npy,0\n') for x in data]
            [fout.write(fk_dir + '/' + '_'.join([x[1], x[0]])+'.npy,0\n') for x in data]
        
        fout.close()

