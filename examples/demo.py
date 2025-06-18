import jax.numpy as jnp
from jax_raft import raft_small
from PIL import Image
import numpy as np


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255. * 2. - 1.  # normalize to [-1, 1]
    return jnp.array(arr)[None]  # [1, H, W, C]


def main():
    # Load two images
    image1 = load_image("frame1.png")
    image2 = load_image("frame2.png")

    # Initialize model and pre-trained parameters
    raft, variables = raft_small(pretrained=True)

    # Run inference
    flow_predictions = raft.apply(variables, image1, image2, train=False)
    flow = flow_predictions[-1]

    print("Flow shape:", flow.shape)
    # Optional: visualize or save output


if __name__ == "__main__":
    main()
