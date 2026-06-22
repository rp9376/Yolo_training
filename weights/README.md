# Base weights

Pretrained base weights (`.pt`) used to initialize training when you choose
**Pretrained** initialization in the Queue builder. The actual `.pt` files are
git-ignored (they are large); this directory and its README are tracked.

## Naming convention

The app discovers weights by family + size:

| Family   | Filenames                                      |
|----------|------------------------------------------------|
| YOLOv26  | `yolo26n.pt` `yolo26s.pt` `yolo26m.pt` `yolo26l.pt` `yolo26x.pt` |
| YOLOv8   | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt` |

If a weight file is **present**, the matching *Pretrained* option is enabled in
the UI. If it is **absent**, only *From scratch* (training from the model's
`.yaml` architecture) is available for that size — no download is performed.

`scripts/setup.sh` seeds `yolo26n.pt` here from `legacy/yolov26/` so the offline
smoke test and pretrained-nano option work out of the box. Add more weights by
copying the appropriately named `.pt` file into this folder.
