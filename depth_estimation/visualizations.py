from PIL import Image
from os import listdir, path
import torch.nn.functional as F
import torchvision.transforms.functional as T
from depth_estimation_module import DepthEstimationModule
import rerun as rr


def camera_intrinsics(image):
    w = image.size(0)
    h = image.size(1)
    v_cen = w / 2
    u_cen = h / 2
    f_len = w * 0.7
    return [[f_len, 0, u_cen], [0, f_len, v_cen], [0, 0, 1]]


model = DepthEstimationModule.load_from_checkpoint(
    "models/best_model.ckpt", map_location="cpu"
)
model.eval()

rr.init("Depth Estimation", spawn=True)
rr.log_view_coordinates("world", up="-Y", timeless=True)
rr.log_view_coordinates("world/camera", xyz="RDF")

base = "data/nyu2_train/"
for location in listdir(base)[:10]:
    location_path = path.join(base, location)
    for fp in listdir(location_path)[:10]:
        fp = path.join(location_path, fp)
        fp = path.splitext(fp)[0]
        img = Image.open(f"{fp}.jpg").convert("RGB")
        tgt = Image.open(f"{fp}.png").convert("L")
        x = T.to_tensor(T.resize(img, (224, 224)))
        y = T.to_tensor(T.resize(tgt, (224, 224))).permute((1, 2, 0))
        e = (
            F.interpolate(model(x.unsqueeze(0)), (224, 224))
            .squeeze(0)
            .permute((1, 2, 0))
        )
        rr.log_pinhole(
            "world/camera",
            child_from_parent=camera_intrinsics(e),
            width=e.size(0),
            height=e.size(1),
        )
        rr.log_image("world/camera/image/rgb", x.squeeze().permute((1, 2, 0)))
        rr.log_depth_image("world/camera/image/depth", y)
        rr.log_depth_image("world/camera/image/estimated", e)
