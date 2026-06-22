"""Framework-agnostic engine for YOLO Training Studio.

This package must not import FastAPI and must lazy-import ultralytics/torch
(only inside training/export functions) so the API and dashboard boot even if
the training stack is broken.
"""
