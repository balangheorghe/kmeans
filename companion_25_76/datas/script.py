import scipy.misc as scm

bytes = scm.imread("bird_small.tiff")

handler = open("points", "w")
for item in bytes:
    for tt in item:
        handler.write("{} {} {}\n".format(tt[0], tt[1], tt[2]))
handler.close()

