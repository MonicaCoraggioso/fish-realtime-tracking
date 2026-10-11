# Hardware

| Component | Role |
|---|---|
| FLIR / Point Grey camera (Spinnaker SDK, via `rotpy`) | 250 fps capture, 608×600 px region of interest |
| Carbide 3D **Shapeoko** CNC (GRBL firmware, G-code over USB serial) | Motorised gantry that moves the camera to follow the fish |
| Projector or second screen (PsychoPy) | Closed-loop visual stimuli (textured backgrounds) |
| Windows PC with an NVIDIA GPU | Runs acquisition, tracking and YOLOv8 inference |

- [`scheme_setup.pdf`](scheme_setup.pdf): schematic of the set-up.
- [`shapeoko/`](shapeoko/): GRBL default settings and a G-code cheat sheet used to configure the gantry ([GRBL wiki](https://github.com/gnea/grbl/wiki)).
- Camera documentation: see the FLIR FlyCapture / Spinnaker manuals on the manufacturer's website.
