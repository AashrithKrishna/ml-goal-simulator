# ⚽ Goal Probability Simulator

A working AI-powered system that analyzes football/soccer highlight clips to predict goal probability using Expected Goals (xG) methodology.

## 🎯 What It Does

1. **Upload a highlight clip** showing a shot on goal
2. **AI analyzes the video** to detect players, ball, and goal
3. **Calculates real xG** based on distance, angle, defensive pressure, and shot type
4. **Displays the probability** of the goal being scored

## 🚀 Quick Start

### Option 1: Run Demo Immediately (No Setup)

```bash
python simple_demo.py
```

This runs a CLI demo showing xG calculations for various shot scenarios. No dependencies needed to see it work!

### Option 2: Full Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
├── xg_calculator.py    # Core xG calculation logic (real sports analytics formulas)
├── video_analyzer.py  # Computer vision using YOLOv8 for object detection
├── app.py             # Streamlit web application
├── simple_demo.py     # CLI demo (run without dependencies)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🔬 How It Works

### The xG Formula

The Expected Goals (xG) model uses multiple factors:

```
xG = Base_xG × Shot_Quality × Pressure_Factor × Keeper_Factor
```

#### 1. Distance Factor
- Exponential decay model
- Most goals scored within 20 meters
- Very rare beyond 30 meters

#### 2. Angle Factor
- Wider angle to goal = easier shot
- Calculated using vector mathematics

#### 3. Defensive Pressure
- More defenders between ball and goal = lower xG
- Effect is stronger close to goal

#### 4. Shot Quality Multipliers
| Factor | Multiplier |
|--------|------------|
| Right foot | 1.00 |
| Left foot | 0.95 |
| Header | 0.50 |
| Counter attack | +15% |
| Open play | 1.00 |
| Penalty | 3.50 |
| Free kick | 0.70 |
| Corner | 0.80 |

## 🎮 Example Results

| Scenario | Distance | xG | Probability |
|----------|----------|-----|-------------|
| Penalty | 11m | 0.85 | 85% |
| Tap-in | 5m | 0.78 | 78% |
| Counter attack | 14m | 0.45 | 45% |
| Long range | 28m | 0.04 | 4% |
| Header from corner | 12m | 0.18 | 18% |

## 🧠 The Science Behind xG

xG (Expected Goals) is used by:
- **Top clubs**: Liverpool, Manchester City, Barcelona
- **Broadcasters**: Sky Sports, ESPN, BT Sport
- **Betting companies**: For odds calculation

The model is trained on millions of historical shots and their outcomes.

## 📊 For Your College Portfolio

This project demonstrates:

1. **Computer Vision**: YOLOv8 object detection, video frame analysis
2. **Machine Learning**: xG prediction using statistical models
3. **Data Science**: Real sports analytics methodology
4. **Web Development**: Streamlit UI/UX
5. **Problem Solving**: Converting raw video → meaningful prediction

### Suggested Research Paper Structure:

```
1. Abstract
   - Introduce xG concept and goal prediction
   - State objectives and methodology

2. Introduction  
   - Background on sports analytics
   - Problem statement: can we predict goals from video?

3. Related Work
   - xG history and adoption in football
   - Computer vision in sports

4. Methodology
   - Distance and angle calculation
   - Defensive pressure modeling
   - Shot quality factors

5. Results
   - Test scenarios and xG outputs
   - Comparison with real-world data

6. Conclusion
   - Achievements and limitations
   - Future improvements

7. References
   - Rudd, S. (2011) - Opta Sports Analytics
   - YOLO paper (Redmon et al.)
```

## 🔧 Customization

### Add Your Own Video Analysis

```python
from video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer()
results = analyzer.analyze_video('your_video.mp4', max_frames=100)
print(results)
```

### Use Calculator Directly

```python
from xg_calculator import xGCalculator

calc = xGCalculator()
result = calc.calculate_xg(
    shot_x=97,           # Distance from goal
    shot_y=30,           # Position on pitch
    defenders_in_path=1, # Number of blockers
    body_part='head',    # head, right_foot, left_foot, other
    shot_type='corner'   # open_play, penalty, free_kick, etc.
)

print(f"xG: {result['xG']}")
print(result['interpretation'])
```

## ⚠️ Limitations & Future Improvements

**Current:**
- Simplified distance estimation from video
- Basic angle calculation
- Heuristic-based defensive pressure

**Future (if you had more time):**
- [ ] Train custom model on football-specific data
- [ ] Use homography for accurate pitch coordinates
- [ ] Track player movements across frames
- [ ] Identify goalkeeper position
- [ ] Classify shot type automatically (volley, header, etc.)

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Understanding xG](https://www.optasports.com/news/introducing-xg/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

Made for college portfolio project ⚽
