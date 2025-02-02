#!/bin/sh
gunicorn app:server --bind=0.0.0.0:8000
