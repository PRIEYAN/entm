"""Serving stage: CT2 int8 translation (marian_ct2) + Piper TTS (speak).

Both are imported by the Jetson voice pipeline (nvidia/engine.py). The old Pi
FastAPI service and the standalone translate CLI were removed on the
jetson-realtime branch — the socket server in nvidia/ replaces them.
"""
