# Architecture

Browser UI (HTML/CSS/JavaScript) → Flask REST API → scikit-learn preprocessing pipeline + Random Forest classifier → SQLite assessment history.

Training data is generated synthetically at first run for reproducible demonstration. The current prototype has no live CCTV, IoT, weather service, authentication, or automatic emergency dispatch integration.
