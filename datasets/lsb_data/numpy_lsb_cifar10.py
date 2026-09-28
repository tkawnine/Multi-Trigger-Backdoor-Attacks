
import pickle
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_BATCH = "test_batch" #CHANGE THE BATCH INPUT FILE NAME

OUTPUT_BATCH = "test_batch_lsb" #CHANGE THE BATCH OUTPUT FILE NAME

BITS_PER_PIXEL = 1


# ============================================================
# LOAD CIFAR-10
# ============================================================

print("Loading CIFAR-10...")

with open(INPUT_BATCH, "rb") as f:
    batch = pickle.load(f, encoding="bytes")


images = batch[b"data"]
labels = batch[b"labels"]
filenames = batch[b"filenames"]


print("Images:", len(images))
print("Image shape:", images.shape)


# ============================================================
# COPY DATA
# ============================================================

stego_images = images.copy()


# ============================================================
# CREATE PAYLOAD
# ============================================================

# This is the information we want to embed.
message = b"LSB-CIFAR10"

# Convert message to bits
payload_bits = np.unpackbits(
    np.frombuffer(message, dtype=np.uint8)
)


print("Payload:", message)
print("Payload bits:", len(payload_bits))


# ============================================================
# EMBED LSB
# ============================================================

for i in range(len(stego_images)):

    image = stego_images[i]

    # Make a flat array of all 3072 pixel values
    flat = image.flatten()

    # Make sure payload fits
    if len(payload_bits) > len(flat):
        raise ValueError(
            "Payload is too large for CIFAR-10 image"
        )

    # Clear the LSB
    flat[:len(payload_bits)] &= 254

    # Insert payload bits
    flat[:len(payload_bits)] |= payload_bits

    # Put back into CIFAR format
    stego_images[i] = flat.reshape(3072)

    if (i + 1) % 1000 == 0:
        print(
            f"Processed {i + 1}/{len(stego_images)}"
        )


# ============================================================
# CREATE OUTPUT BATCH
# ============================================================

output_batch = {
    b"data": stego_images,
    b"labels": labels,
    b"filenames": filenames
}


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_BATCH, "wb") as f:
    pickle.dump(output_batch, f)


print()
print("==============================")
print("LSB STEGANOGRAPHY COMPLETE")
print("==============================")
print("Original:", INPUT_BATCH)
print("Stego   :", OUTPUT_BATCH)
print("Shape   :", stego_images.shape)
print("Labels preserved:", np.array_equal(
    labels,
    output_batch[b"labels"]
))

