from matplotlib import pyplot as plt
import processing.pipeline as p
import processing.transformations.segmentation as seg

r1 = p.Reader(seg.file_paths_ordered("/home/felix/Documents/data", iterations=2), name="Read File Names")
t1 = r1.transform(seg.numpy_load_slice(), name="Load Volume")
t2 = t1.transform(seg.fuse_labels_greater_than(2), name="Fuse Labels")
t3 = t2.transform(seg.numpy_clip(-100, 400), name="Clip Pixel Values")
t4 = t3.transform(seg.numpy_to_simpleITK(), name="To SimpleITK")
t5 = t4.transform(seg.simpleITK_rescale_intensitiy(), name="Scale Pixel Values")
# t6 = t5.transform(seg.simpleITK_scale([64, 128, 128], [4, 2, 2]), name="Scale Volumes")
t6 = t5.transform(seg.simpleITK_scale([128, 128], [2, 2]), name="Scale Volumes")
ta = t6.transform(seg.simpleITK_deform(3, 50), name="Deform Volumes")
t7 = ta.transform(seg.simpleITK_to_numpy(), name="To NumPy")
t8 = t7.transform(seg.numpy_transpose(), name="Transpose")
t9 = t8.combine(t8, seg.numpy_histogram_matching(), name="Match Histograms")
last = t9.run_on(2)

print
print last
print

for input_tuple in last:

    inputs = input_tuple[0]
    parameters = input_tuple[1]

    img = inputs[0]
    lbl = inputs[1]

    # middle = img.shape[0] / 2

    print parameters

    plt.figure(1)
    # plt.imshow(np.squeeze(img[middle, :, :]), interpolation='nearest', cmap='gray')
    plt.imshow(img, interpolation='nearest', cmap='gray')

    plt.figure(2)
    # plt.imshow(np.squeeze(lbl[middle, :, :]), interpolation='nearest', cmap='gray')
    plt.imshow(lbl, interpolation='nearest', cmap='gray')

    plt.show()
