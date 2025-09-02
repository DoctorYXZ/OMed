import argparse
import numpy as np
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None  # Allowing any large images

def normalizeStaining(img,img_name, saveNorm=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearance of H&E stained images '''
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log((img.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # Check if ODhat has enough data
    if len(ODhat) == 0:
        print(f"Warning: No valid optical density values after filtering for image: {img_name}. Skipping image.")
        return None

    if ODhat.shape[0] < 2:  # Ensure there are at least 2 valid data points
        print(f"Warning: Insufficient data for covariance calculation for image: {img_name}. Skipping image.")
        return None

    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    if saveNorm is not None:
        Image.fromarray(Inorm).save(saveNorm)

    return Inorm

def process_folder(input_folder, output_folder):
    ''' Process all HE patches in the specified folder '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.tif'):
            img_path = os.path.join(input_folder, file_name)

            # Use context manager to ensure that image files are closed in a timely manner
            with Image.open(img_path) as img:
                img = np.array(img)
                height, width, _ = img.shape


                normalized_img = normalizeStaining(img, file_name)

                if normalized_img is not None:  # Ensure that we only save normalized images
                    # Save the normalized image
                    output_path = os.path.join(output_folder, file_name)
                    Image.fromarray(normalized_img).save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFolder', type=str,  default='C:\\path\\to\\your 224patch files',
                        help='Input folder containing H&E patches (tif files)')
    parser.add_argument('--outputFolder', type=str,  default='C:\\path\\to\\save your color-normalized 224patch files', help='Output folder to save color-normalized patches')
    parser.add_argument('--Io', type=int, default=240, help='Transmitted light intensity')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha for calculation')
    parser.add_argument('--beta', type=float, default=0.15, help='Beta for calculation')

    args = parser.parse_args()

    process_folder(args.inputFolder, args.outputFolder)
