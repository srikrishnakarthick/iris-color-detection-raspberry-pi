---

## Setup & Execution

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install system dependencies (Raspberry Pi)

```bash
sudo apt update
sudo apt install -y cifs-utils network-manager python3-tk
```

### 3. Setup LAN transfer

```bash
bash scripts/setup_lan.sh
```

### 4. Run image capture (on Raspberry Pi)

```bash
python src/button.py
```

### 5. Run analysis (on laptop)

```bash
python src/iris_analysis.py
```

---

## Results

### 📷 Captured Image

![Captured](results/0.2026-03-4_15-44-11.jpg)

### Iris Detection

![Iris Detection](results/1.Iris_detection.png)

### K-means Clustering

![Clustering](results/2.Clustering.png)

### Histograms (RGB & HSV)

![Histograms](results/3.Histograms.png)

### Final Output

![Summary](results/4.Summary.png)

---

## Observations

* HSV color space provides **better robustness to lighting variations** than RGB
* K-means clustering effectively identifies dominant iris color (mean RGB value needn't reflect dominant eye color)
* Removing sclera and glare significantly improves clustering accuracy
* Demonstrated successfully to ~350 visitors; darker eye colours were predominant among the Indian population sampled

---

## Acknowledgements

Thanks to **Dr Prasanna Katti** (Principal Investigator, Muscle Physiology Lab) and the **Science Day Committee at IISER Tirupati** for supporting and funding this project.

---

## 📎 Notes

* Ensure shared Windows folder is mounted before capture
* Replace IP addresses and credentials in setup script
