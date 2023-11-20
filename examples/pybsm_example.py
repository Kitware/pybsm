"""
This script is an introductory example of how to use pyBSM.

This script also includes an example of using pyBSM to simulate imagery.
Be sure that the test image is in the same folder as this script.
"""
import numpy as np
import matplotlib.pyplot as plt
import pybsm.simulation as simulation


img_fname = (
    "./examples/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"
)


img = plt.imread(img_fname)  # cv2.imread(img_fname)[:, :, ::-1]

ref_img = simulation.RefImage(img, 0.3)
ref_img.show()


# This should be a no-op.
sensor, scenario = ref_img.estimate_capture_parameters(altitude=10000)
img_out = simulation.simulate_image(ref_img, sensor, scenario)[2]
img_out = simulation.stretch_contrast_convert_8bit(img_out)

plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.imshow(simulation.stretch_contrast_convert_8bit(ref_img.img), cmap="gray")
ax2 = plt.subplot(1, 2, 2)
plt.imshow(img_out, cmap="gray")
plt.show()

# telescope focal length (m)
f = 4
# Telescope diameter (m)
D = 275e-3

# detector pitch (m)
p = 0.008e-3

# Optical system transmission, red  band first (m)
optTransWavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6

ihaze = 1

altitude = 9000

ground_range = 0.0

scenario = simulation.Scenario("virtual_camera", ihaze, altitude, ground_range)

sensor = simulation.Sensor("virtual_camera", D, f, p, optTransWavelengths)

ground_range = np.arange(0, 101e3, 10e3)

ref_img.pix_values = np.array([ref_img.img.min(), ref_img.img.max()])
ref_img.refl_values = np.array([0.05, 0.5])
ref_img.gsd = 3.19 / 160.0
idx = 1
plt.subplot(2, 4, 1)
plt.title("Input image", fontdict={"fontsize": 8})
plt.imshow(simulation.stretch_contrast_convert_8bit(ref_img.img), cmap="gray")
for ii in np.arange(4, ground_range.shape[0]):
    idx += 1
    plt.subplot(2, 4, idx)
    plt.suptitle(
        "ihaze-" + str(ihaze) + "_altitude-" + str(altitude), fontsize=10
    )
    plt.title(
        "groundRange: " + str(ground_range[ii]) + " km",
        fontdict={"fontsize": 6},
    )
    scenario.ground_range = ground_range[ii]
    img_out = simulation.simulate_image(ref_img, sensor, scenario)[2]
    img_out = simulation.stretch_contrast_convert_8bit(img_out)
    plt.imshow(img_out, cmap="gray")
plt.savefig(
    "./examples/refactor_branch/fig_ihaze-"
    + str(ihaze)
    + "_alt-"
    + str(altitude)
    + ".png"
)
