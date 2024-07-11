# created by ilayd on 06/07/2024
import torch
import warnings
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes

# suppress specific warning
warnings.filterwarnings("ignore", message="No mtl file provided")

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the 3D model (mesh)
obj_filename = "lpwkull_2.obj"
mesh = load_objs_as_meshes([obj_filename], device=device)

# debug: print mesh details
print(f"Loaded mesh with {mesh.verts_list()[0].shape[0]} vertices and {mesh.faces_list()[0].shape[0]} faces")

# print out some vertices for debugging
print("First 5 vertices:")
print(mesh.verts_list()[0][:5])

# scale and center the mesh
verts = mesh.verts_list()[0]
verts_centered = verts - verts.mean(0)
scale = verts_centered.abs().max()
verts_scaled = verts_centered / scale

verts_rgb = torch.ones_like(verts_scaled)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

mesh = Meshes(verts=[verts_scaled.to(device)], faces=mesh.faces_list(), textures=textures)

# initialize the camera with different elevation and azimuth
R, T = look_at_view_transform(2.7, 20, 45)  # distance, elevation, azimuth
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# add a single light source
lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

# debug: print light details
print(f"Light location: {lights.location}")

# initialize the rasterizer and shader
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

# render the mesh
images = renderer(mesh)

# convert the image to numpy and display it
image = images[0, ..., :3].cpu().numpy()
plt.imshow(image)
plt.axis("off")
plt.show()
