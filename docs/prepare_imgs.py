import json
import os
import random
from glob import glob
from shutil import copyfile

source_dirs_by_step = {
    'test': [
        ('../data-livdet-2015/Testing/Digital_Persona/Fake', ['Ecoflex', 'Gelatine', 'Latex', 'Liquid_Ecoflex', 'RTV', 'WoodGlue']),
        ('../data-livdet-2015/Testing/Digital_Persona/Live', None),
    ],
    'train': [
        ('../data-livdet-2015/Training/Digital_Persona/Fake', ['Ecoflex', 'Gelatine', 'Latex', 'WoodGlue']),
        ('../data-livdet-2015/Training/Digital_Persona/Live', None),
    ]
}
target_dir = 'quiz/img'

res = {'test': {'fake': [], 'live': []}, 'train': {'fake': [], 'live': []}}
for step, source_dirs in source_dirs_by_step.iteritems():
    for d, methods in source_dirs:
        is_fake = methods is not None
        N_FILES = 25 if is_fake else 100
        if not is_fake:
            methods = ['']

        for method in methods:
            files = glob(d + '/' + method + '*.png')
            random.shuffle(files)
            for f in files[:N_FILES]:
                k = 'fake' if is_fake else 'live'
                new_d = target_dir + '/{:s}/{:s}'.format(step, k)
                if not os.path.isdir(new_d):
                    os.makedirs(new_d)
                new_f = new_d + '/' + os.path.basename(f)

                res[step][k].append(new_f.replace('quiz/', ''))
                copyfile(f, new_f)

with open('quiz/list.json', 'w') as f:
    f.write('var LIST = ' + json.dumps(res, indent=4) + ';\n')
