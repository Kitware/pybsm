"""This script is an introductory example of how to use pyBSM.

This script also includes an example of using pyBSM to simulate imagery.
Be sure that the test image is in the same folder as this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pybsm.simulation as simulation

asset_dir = Path(__file__).parent

img_file_name = str(
    asset_dir / "data" / "M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"
)


img = plt.imread(img_file_name)  # cv2.imread(img_file_name)[:, :, ::-1]

# the width of the tank is 319 cm and it spans ~160 pixels in the image
ref_img = simulation.RefImage(img, gsd=(3.19 / 160.0))
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
opt_trans_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6

ihaze = 1

altitude = 9000

ground_range = 0.0

scenario = simulation.Scenario("virtual_camera", ihaze, altitude, ground_range)

sensor = simulation.Sensor("virtual_camera", D, f, p, opt_trans_wavelengths)

ground_ranges = np.arange(0, 101e3, 10e3)

ref_img.pix_values = np.array([ref_img.img.min(), ref_img.img.max()])
ref_img.refl_values = np.array([0.05, 0.5])
idx = 1
plt.subplot(2, 4, 1)
plt.title("Input image", fontdict={"fontsize": 8})
plt.imshow(simulation.stretch_contrast_convert_8bit(ref_img.img), cmap="gray")
for ii in np.arange(4, ground_ranges.shape[0]):
    idx += 1
    plt.subplot(2, 4, idx)
    plt.suptitle("ihaze-" + str(ihaze) + "_altitude-" + str(altitude), fontsize=10)
    plt.title(
        "groundRange: " + str(ground_ranges[ii] / 1000) + " km",
        fontdict={"fontsize": 6},
    )
    scenario.ground_range = ground_ranges[ii]
    img_out = simulation.simulate_image(ref_img, sensor, scenario)[2]
    img_out = simulation.stretch_contrast_convert_8bit(img_out)
    plt.imshow(img_out, cmap="gray")
plt.show()
