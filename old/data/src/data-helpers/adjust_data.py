def adjust_data(folder_i, folder_o, init, final):
    for i in range(init, final):
        pathi = "./../../data/"+folder_i+"/data-"+str(i)+".csv"
        patho ="./../../data/"+folder_o+"/data-"+str(i)+".csv"
        with open(pathi, "r") as fin:
            with open(patho, "w") as fout:
                for line in fin:
                    l = line
                    l = l.replace('"[', '')
                    l = l.replace(']"', '')
                    l = l.replace('\'', '')
                    fout.write(l)

def mix_data(init, final):
    for i in range(init, final):
        pathi_gt = "./../../data/final-gt/data-"+str(i)+".csv"
        pathi ="./../../data/final/data-"+str(i)+".csv"
        patho = "./../../data/final-mix/data-" + str(i) + ".csv"
        patho_dirty = "./../../data/dirty-mix/data-" + str(i) + ".csv"
        with open(pathi, "r") as fin:
            with open(patho_dirty, "w") as fout:
                for line in fin:
                    l = line[:-4]
                    l+=", \n"
                    fout.write(l)

        with open(pathi_gt) as xh:
            with open(patho_dirty) as yh:
                with open(patho, "w") as zh:
                    # Read first file
                    xlines = xh.readlines()
                    # Read second file
                    ylines = yh.readlines()
                    # Combine content of both lists  and Write to third file
                    for line1, line2 in zip(ylines, xlines):
                        zh.write("{} {}\n".format(line1.rstrip(), line2.rstrip()))

        fin.close()
        fout.close()

def adjust_all_datasets():
    adjust_data("dirty", "final", 1, 31)
    adjust_data("dirty-gt", "final-gt", 1, 31)
    mix_data(1, 31)