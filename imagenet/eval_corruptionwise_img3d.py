from glob import glob
import numpy as np
import re

ordered_corr = ['near_focus', 'far_focus', 'bit_error', 'color_quant', 
                   'flash', 'fog_3d', 'h265_abr', 'h265_crf',
                   'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur'] # 12 corruptions in ImageNet-3DCC


def read_file(filename):
    lines = open(filename, "r").readlines()
    acc = []
    bri = []
    nll = []
    corr_name_all = []
    for line in lines:
        if "error : " in line:
            # print(line.split(":")[-3].split(",")[0].strip())
            corr_name = line.split(" ")[6][1:-1].split("_")[0]
            corr_name = re.sub(r'\d+', '', corr_name)
            # assert False
            acc.append(float(line.split(":")[-3].split(",")[0].strip()[:-1]))
            bri.append(float(line.split(":")[-2].split(",")[0].strip()))
            nll.append(float(line.split(":")[-1].strip()))
            corr_name_all.append(corr_name)
    assert len(acc)==len(ordered_corr)==12
    # print(corr_name_all)
    return np.mean(np.array(acc)), np.mean(np.array(bri)), np.mean(np.array(nll))


def read_file_corr(filename, acc, bri, nll):
    lines = open(filename, "r").readlines()
    # corr_name_all = []
    for line in lines:
        if "error : " in line:
            # print(line.split(":")[-3].split(",")[0].strip())
            # print("line:",line)
            # print("line.split(" ")[6][1:-1]:",line.split(" ")[6][1:-1])
            # corr_name = line.split(" ")[6][1:-1].split("_")[0]
            corr_name = line.split(" ")[6][1:-2]#.split("_")[0]
            # print("corr_name:",corr_name)
            # corr_name = re.sub(r'\d+', '', corr_name)
            # assert False
            if corr_name in acc:
                acc[corr_name].append(float(line.split(":")[-3].split(",")[0].strip()[:-1]))
                bri[corr_name].append(float(line.split(":")[-2].split(",")[0].strip()))
                nll[corr_name].append(float(line.split(":")[-1].strip()))
            else:
                acc[corr_name] = [float(line.split(":")[-3].split(",")[0].strip()[:-1])]
                bri[corr_name] = [float(line.split(":")[-2].split(",")[0].strip())]
                nll[corr_name] = [float(line.split(":")[-1].strip())]
                # corr_name_all.append(corr_name)
    # print(corr_name_all)
    # print(acc)
    assert len(acc)==len(ordered_corr)==12
    return acc, bri, nll


def read_files(files):
    
    acc = {}
    bri = {}
    nll = {}
    if len(files) == 1:
        for f in files:
            acc, bri, nll = read_file_corr(f, acc, bri, nll)
            # accs.append(acc)
            # bris.append(bri)
            # nlls.append(nll)
        print("read", len(files), "files.")
        for key in acc:
            acc[key] = acc[key][0]
        # print(acc)
        # print("Brier")
        for key in bri:
            bri[key] = bri[key][0]
        # print(bri)
        # print("NLL")
        for key in nll:
            nll[key] = nll[key][0]
        # res={}
        # res["acc"] = [np.mean(np.array(accs)), np.std(np.array(accs))]
        # res["bri"] = [np.mean(np.array(bris)), np.std(np.array(bris))]
        # res["nll"] = [np.mean(np.array(nlls)), np.std(np.array(nlls))]
        return acc, bri, nll
    else:

        for f in files:
            acc, bri, nll = read_file_corr(f, acc, bri, nll)
            # accs.append(acc)
            # bris.append(bri)
            # nlls.append(nll)
        print("read", len(files), "files.")
        # print("Error")
        for key in acc:
            acc[key] = np.mean(np.array(acc[key]))
        # print(acc)
        # print("Brier")
        for key in bri:
            bri[key] = np.mean(np.array(bri[key]))
        # print(bri)
        # print("NLL")
        for key in nll:
            nll[key] = np.mean(np.array(nll[key]))
        # print(nll)
        
        return acc, bri, nll
        # res={}
        # res["acc"] = [np.mean(np.array(accs)), np.std(np.array(accs))]
        # res["bri"] = [np.mean(np.array(bris)), np.std(np.array(bris))]
        # res["nll"] = [np.mean(np.array(nlls)), np.std(np.array(nlls))]


print("Reading ImageNet3DCC petalfim files...")
# result = read_files(glob("cotta[0-9]_*.txt"))
acc, bri, nll = read_files(glob("output/imagenet3d/petalfim/petalfim[0-9]_*.txt"))


print("\nError")
for c in ordered_corr:
    print(c + "," + str(round(acc[c],2)))

print("\nBrier")
for c in ordered_corr:
    print(c + "," + str(round(bri[c],4)))

print("\nNLL")
for c in ordered_corr:
    print(c + "," + str(round(nll[c],4)))


# for key in result:
# print("Error:", "Mean:",result["acc"][0], "Std:",result["acc"][1])
# print("Brier:", "Mean:",result["bri"][0], "Std:",result["bri"][1])
# print("NLL:", "Mean:",result["nll"][0], "Std:",result["nll"][1])
# print(result)
