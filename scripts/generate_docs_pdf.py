import os
import sys
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to calculate the total page count dynamically
    and draw consistent headers, footers, and page numbers on all pages except the cover page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # We suppress headers and footers on the cover page (Page 1)
        if self._pageNumber == 1:
            return
        
        self.saveState()
        
        # Color palette
        primary_color = HexColor("#1e293b")  # Dark Slate
        divider_color = HexColor("#cbd5e1")  # Light gray divider
        text_color = HexColor("#64748b")     # Muted Slate
        
        # Header (Top of Page)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(primary_color)
        self.drawString(54, 750, "IPL MATCH SIMULATION SYSTEM")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(text_color)
        self.drawRightString(558, 750, "Technical System Architecture & Codebase Overview")
        
        # Header line
        self.setStrokeColor(divider_color)
        self.setLineWidth(0.75)
        self.line(54, 742, 558, 742)
        
        # Footer (Bottom of Page)
        self.line(54, 52, 558, 52)
        
        self.setFont("Helvetica", 8)
        self.drawString(54, 40, "Confidential - Project Documentation")
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_str)
        
        self.restoreState()

def html_escape(text: str) -> str:
    """Escapes standard HTML characters to prevent ReportLab markup parsing crashes."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def main():
    pdf_filename = "IPL_Simulation_System_Documentation.pdf"
    
    # Establish document dimensions with 0.75-inch (54 pt) side margins,
    # and 1-inch (72 pt) top/bottom margins to separate header/footer elements.
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    styles['Normal'].textColor = HexColor("#334155")
    styles['Normal'].fontSize = 9.5
    styles['Normal'].leading = 13.5
    
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=HexColor("#1e293b"),
        alignment=0,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=HexColor("#475569"),
        alignment=0,
        spaceAfter=40
    )
    
    meta_label_style = ParagraphStyle(
        'CoverMetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=HexColor("#1e293b"),
    )
    
    meta_val_style = ParagraphStyle(
        'CoverMetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=HexColor("#475569"),
    )
    
    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=HexColor("#1e293b"),
        spaceBefore=22,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=HexColor("#2563eb"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CodeBlockCustom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=HexColor("#0f172a"),
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 100))
    
    # Decorative accent bar
    accent_bar = Table([[""]], colWidths=[504], rowHeights=[6])
    accent_bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#3b82f6")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(accent_bar)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("IPL MATCH SIMULATION &<br/>TEAM OPTIMIZATION SYSTEM", title_style))
    story.append(Paragraph("Technical Codebase Blueprint, System Design Document & File Explanations", subtitle_style))
    story.append(Spacer(1, 150))
    
    # Cover Metadata Block
    meta_data = [
        [Paragraph("Author:", meta_label_style), Paragraph("Antigravity Coding Assistant (DeepMind)", meta_val_style)],
        [Paragraph("Target Audience:", meta_label_style), Paragraph("Engineering and Development Team", meta_val_style)],
        [Paragraph("Date of Compilation:", meta_label_style), Paragraph(datetime.now().strftime("%B %d, %Y"), meta_val_style)],
        [Paragraph("Document Status:", meta_label_style), Paragraph("Approved Technical Reference", meta_val_style)],
        [Paragraph("System Version:", meta_label_style), Paragraph("1.0.0 (FastAPI Backend + React Frontend)", meta_val_style)],
    ]
    meta_table = Table(meta_data, colWidths=[130, 374])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, HexColor("#f1f5f9")),
    ]))
    story.append(meta_table)
    
    story.append(PageBreak())

    # Helper function to generate clean code-like blocks
    def code_box(code_text: str) -> Table:
        escaped = html_escape(code_text)
        formatted = escaped.replace("\n", "<br/>").replace(" ", "&nbsp;")
        p = Paragraph(formatted, code_style)
        t = Table([[p]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 0.5, HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    # Helper function to generate file subsections
    def file_section(file_path: str, purpose: str, features: list, code_snippet: str = None):
        elements = [
            Paragraph(f"📄 {file_path}", h2_style),
            Paragraph(f"<b>Core Purpose:</b> {purpose}", body_style)
        ]
        
        # Add Bullet Points
        bullet_intro = Paragraph("<b>Key Responsibilities &amp; Implementations:</b>", body_style)
        elements.append(bullet_intro)
        for f in features:
            elements.append(Paragraph(f"• {f}", bullet_style))
            
        # Add Code Snippet if present
        if code_snippet:
            elements.append(Spacer(1, 4))
            elements.append(code_box(code_snippet))
            
        elements.append(Spacer(1, 8))
        return KeepTogether(elements)

    # =========================================================================
    # SECTION 1: ARCHITECTURE OVERVIEW
    # =========================================================================
    story.append(Paragraph("1. System Architecture &amp; Workflow", h1_style))
    story.append(Paragraph(
        "This project is an end-to-end Machine Learning-powered system designed to simulate individual cricket matches "
        "at a delivery-by-delivery level and optimize team lineups. It uses historical Indian Premier League (IPL) data, "
        "applies robust feature engineering, trains a multi-class predictive model, runs parallel Monte Carlo simulations, "
        "and optimizes playing XIs based on credits, roles, and historical/simulated player projections.",
        body_style
    ))
    
    # Diagram/workflow representation
    flow_data = [
        [Paragraph("<b>Step 1: Data Preprocessing</b><br/>Cleans raw CSV and computes historical averages &amp; running stats.", body_style)],
        [Paragraph("<b>Step 2: Predictive Modeling</b><br/>Trains an XGBoost classifier in Google Colab to predict delivery-level outcomes.", body_style)],
        [Paragraph("<b>Step 3: Stochastic Simulation</b><br/>Simulates matches ball-by-ball. Aggregates results in a Monte Carlo loop.", body_style)],
        [Paragraph("<b>Step 4: Decision &amp; Selection Optimization</b><br/>Runs linear programming and heuristics to choose teams and rotation plans.", body_style)],
        [Paragraph("<b>Step 5: Interactive Web App</b><br/>Serves simulation settings and renders scoreboard summaries and distributions.", body_style)]
    ]
    flow_table = Table(flow_data, colWidths=[480])
    flow_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))
    story.append(flow_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: DATA CLEANING & PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("2. Data Cleaning &amp; Feature Engineering Pipeline", h1_style))
    story.append(Paragraph(
        "Before feeding features into a model, raw data must be loaded, cleaned, and contextualized. "
        "The scripts in this pipeline convert standard ball-by-ball IPL dataset sheets into ML-ready historical averages.",
        body_style
    ))

    # File: cleaner.py
    story.append(file_section(
        file_path="src/data/cleaner.py",
        purpose="Standardizes raw deliveries data by removing records with missing critical identifiers and mapping raw outputs to discrete categorical outcomes.",
        features=[
            "Filters out invalid deliveries (e.g., missing strikers, bowlers, or overs) to maintain database type safety.",
            "Parses the custom fractional over format (e.g., 0.1, 0.2 ... 0.6) into numeric overs (0-19) and legal ball numbers (1-6).",
            "Derives legal delivery flags, runs scored off bat, extras conceded, and whether the delivery resulted in a wicket.",
            "Maps outcome states into single string categorical labels ('0', '1', '2', '3', '4', '6', 'W') indicating the result of each delivery.",
            "Categorizes overs into cricket-specific phase labels: 'powerplay' (overs 0-5), 'middle' (overs 6-14), and 'death' (overs 15-19)."
        ],
        code_snippet="def clean(df: pd.DataFrame) -> pd.DataFrame:\n    # Drops nulls, computes wickets, extras, outcomes, and game phases"
    ))

    # File: feature_engineer.py
    story.append(file_section(
        file_path="src/data/feature_engineer.py",
        purpose="Transforms the cleaned match logs into chronological states, establishing rolling and historical parameters for players.",
        features=[
            "Sorts match records sequentially to ensure calculations are chronologically accurate.",
            "Computes cumulative scores, wickets down, and legal balls bowled within an innings to construct real-time match contexts.",
            "Determines second-innings target chasing constraints: target runs, runs needed, and required run rate (RRR) per ball.",
            "Compiles historical batting metrics (career runs, strike rate, boundaries) and bowling metrics (average conceded, economy, wickets) on an expanding window to prevent any data leakage from future events."
        ],
        code_snippet="def build_features(df: pd.DataFrame) -> pd.DataFrame:\n    # Computes cumulative state and career historical aggregates"
    ))

    # File: prepare_data.py
    story.append(file_section(
        file_path="scripts/prepare_data.py",
        purpose="Coordinates the raw loading, cleaning, feature engineering, and saving of artifacts, preparing files for API metadata and Colab model training.",
        features=[
            "Acts as the one-click runner executing both `cleaner.py` and `feature_engineer.py` pipelines in sequence.",
            "Extracts a unique collection of all batters, bowlers, venues, teams, and seasons into `data/processed/meta.json` to feed metadata dropdowns on the frontend web dashboard.",
            "Generates and saves a static `player_stats.json` snapshot containing all historical player averages to serve as a fast cold-start fallback during inference."
        ],
        code_snippet="if __name__ == '__main__':\n    # Runs clean(), build_features(), and exports player_stats.json / meta.json"
    ))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 3: MODEL TRAINING & PREDICTION
    # =========================================================================
    story.append(Paragraph("3. Model Training &amp; Predictor Inference Wrapper", h1_style))
    story.append(Paragraph(
        "The model is a multi-class classifier trained on historical IPL deliveries. It outputs a probability "
        "distribution for the next ball's outcome.",
        body_style
    ))

    # File: colab_training.py
    story.append(file_section(
        file_path="notebooks/colab_training.py (Converted Notebook)",
        purpose="Trains the core XGBoost classifier model on the engineered historical features to predict delivery outcomes.",
        features=[
            "Fits label encoders to categorical features ('striker', 'bowler', 'venue', 'phase') and target labels ('0', '1', '2', '3', '4', '6', 'W').",
            "Implements an XGBClassifier optimized with multi-class log-loss (mlogloss) and CPU histogram tree methods for fast execution.",
            "Evaluates prediction quality on a stratified validation set (accuracy, classification reports, and feature importance).",
            "Saves the trained model (`ipl_ball_model.pkl`), label encoders (`label_encoders.pkl`), and feature list (`feature_columns.pkl`) for real-time inference."
        ],
        code_snippet="model = XGBClassifier(\n    n_estimators=500, max_depth=7, learning_rate=0.05, eval_metric='mlogloss'\n)"
    ))

    # File: predictor.py
    story.append(file_section(
        file_path="src/model/predictor.py",
        purpose="Exposes a unified inference class that loads pickling exports and calculates outcome probabilities under arbitrary player match situations.",
        features=[
            "Wrapper class `BallOutcomePredictor` loading model objects and features during backend start-up.",
            "Injects offline historical career averages for batters/bowlers (`player_stats.json`) if they are missing or hold default placeholders in incoming request payloads.",
            "Applies fitted label encoders to categorical parameters, replacing any unseen players with a default 'Unknown' category to prevent model failures.",
            "Exposes a `MockPredictor` class with static, cricket-heuristic probability arrays, allowing the server to function for testing even when model files are not trained or present."
        ],
        code_snippet="def predict_proba(self, ball_context: dict) -> Dict[str, float]:\n    # Enforces player stat overrides, encodings, and performs model.predict_proba()"
    ))

    # =========================================================================
    # SECTION 4: SIMULATION ENGINE
    # =========================================================================
    story.append(Paragraph("4. Match Simulation &amp; Monte Carlo Engines", h1_style))
    story.append(Paragraph(
        "Rather than using simple average margins, the simulator replicates actual cricket regulations ball-by-ball. "
        "By repeating this stochastic process hundreds of times, we build stable probability distributions.",
        body_style
    ))

    # File: match_simulator.py
    story.append(file_section(
        file_path="src/simulation/match_simulator.py",
        purpose="Simulates a complete T20 match ball-by-ball, handling batsman rotation, bowler overs, wicket-falls, and score progression.",
        features=[
            "Structures entities via dataclasses: `BatterState` (scores, balls, boundaries), `BowlerState` (runs, balls, wickets, extras), and `InningsState` (scores, status tracking).",
            "Replicates actual game logic: batsman strike-rotation (odd-run splits swap strike), bowler limits (maximum overs per bowler), wicket-falls (brings in next order), and innings termination.",
            "Runs stochastic outcome sampling: pulls prediction probabilities from the ML predictor based on the active state and selects an outcome using `random.choices` weight distributions."
        ],
        code_snippet="def simulate(self, team1, team2, batting_order_1, ... ) -> MatchResult:\n    # Coordinates Innings 1, sets target, runs Innings 2, returns full logs"
    ))

    # File: monte_carlo.py
    story.append(file_section(
        file_path="src/simulation/monte_carlo.py",
        purpose="Runs multiple independent matches in parallel to evaluate win probabilities, run distributions, and projected averages.",
        features=[
            "Coordinates execution of N match simulations in parallel using Python's `ThreadPoolExecutor`.",
            "Aggregates outcomes to determine overall win ratios, average margins of victory (runs or wickets), and standard deviations.",
            "Computes statistical confidence margins (e.g., 95% confidence intervals for first/second innings team runs).",
            "Averages out batter and bowler details across all simulations to calculate expected runs, strike rates, wickets, and economy rates."
        ],
        code_snippet="def run_monte_carlo(simulator, team1, team2, ..., n_simulations=1000) -> MonteCarloResult:\n    # Runs N simulations in parallel and invokes _aggregate(results)"
    ))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 5: TEAM OPTIMIZATION
    # =========================================================================
    story.append(Paragraph("5. Playing XI &amp; Fantasy Team Optimizers", h1_style))
    story.append(Paragraph(
        "Using Monte Carlo projections, the optimizer selects the best playing orders and fantasy selections.",
        body_style
    ))

    # File: team_optimizer.py
    story.append(file_section(
        file_path="src/optimization/team_optimizer.py",
        purpose="Implements algorithms for playing orders, bowling assignments, and budget-constrained Dream11 fantasy teams.",
        features=[
            "Defines player role classes: 'BAT' (Batsman), 'BOWL' (Bowler), 'AR' (All-Rounder), and 'WK' (Wicket-Keeper).",
            "Implements `dream11_optimize`: Scores players based on projected fantasy points derived from simulated averages and uses a greedy heuristic to pick the best XI within a 100-credit budget.",
            "Implements `optimize_batting_order`: Orders batters by balancing projected average runs and strike rates, placing aggressive openers first, anchors in the middle, and finishers below.",
            "Implements `optimize_bowling_rotation`: Plans a 20-over bowling rotation by assigning bowlers to phases where their metrics fit best (e.g., death overs assigned to high wicket-taking, low economy bowlers)."
        ],
        code_snippet="def dream11_optimize(players, mc_result, budget=100.0, ...) -> Tuple[List[Player], float]:\n    # Ranks candidates by simulated points and checks constraints (role quotas, overseas limits)"
    ))

    # =========================================================================
    # SECTION 6: FASTAPI REST BACKEND
    # =========================================================================
    story.append(Paragraph("6. REST API Backend Server Layer", h1_style))
    story.append(Paragraph(
        "Exposes all simulation models, Monte Carlo runners, and optimizer methods through FastAPI endpoints.",
        body_style
    ))

    # File: app.py
    story.append(file_section(
        file_path="src/api/app.py",
        purpose="Exposes REST API endpoints and defines Pydantic request models, connecting frontend requests with the simulator.",
        features=[
            "Initializes the FastAPI application and configures CORS (Cross-Origin Resource Sharing) middleware to communicate with the React frontend.",
            "Loads the predictive model (`predictor`) and match simulation engine once at startup to optimize performance.",
            "Serves critical endpoints:\n"
            "  - GET `/meta`: Returns unique batters, bowlers, teams, and venues from dataset.\n"
            "  - POST `/simulate`: Simulates a single match (~50ms response).\n"
            "  - POST `/monte-carlo`: Runs N parallel simulations and returns statistical summaries.\n"
            "  - POST `/optimize/batting-order` &amp; `/optimize/bowling-rotation`: Returns optimized tactics.\n"
            "  - POST `/optimize/dream11`: Resolves credit/budget constraints to select fantasy squads."
        ],
        code_snippet="@app.post('/monte-carlo')\ndef monte_carlo(req: MonteCarloRequest):\n    # Invokes run_monte_carlo() and returns aggregated response keys"
    ))

    # =========================================================================
    # SECTION 7: REACT FRONTEND DASHBOARD
    # =========================================================================
    story.append(Paragraph("7. React Web Application Frontend Layer", h1_style))
    story.append(Paragraph(
        "A React dashboard providing user-friendly controls to configure squads, view simulations, and optimize lineups.",
        body_style
    ))

    # Webapp App.js & CSS
    story.append(file_section(
        file_path="webapp/src/App.js &amp; webapp/src/App.css",
        purpose="Sets up React routing and the global modern visual styling of the dashboard.",
        features=[
            "Configures React Router routing for page-level navigation.",
            "Provides a dark slate visual theme with contrasting cards, responsive grid structures, buttons, and custom layout components."
        ],
        code_snippet="// App.js routes:\n// '/' -> SimulatePage, '/monte-carlo' -> MonteCarloPage, '/optimize' -> OptimizerPage"
    ))

    # Webapp Hooks & Utils
    story.append(file_section(
        file_path="webapp/src/utils/api.js &amp; webapp/src/hooks/useMeta.js",
        purpose="Handles Axios HTTP communications and caches metadata dynamically.",
        features=[
            "Caches unique teams, venues, and player registries from the backend using a custom React hook `useMeta`.",
            "Standardizes Axios payloads mapping directly to FastAPI request specifications."
        ],
        code_snippet="export const runMonteCarlo = async (data) => {\n  const res = await api.post('/monte-carlo', data);\n  return res.data;\n};"
    ))

    # Webapp Components
    story.append(file_section(
        file_path="webapp/src/components/ (TeamBuilder.js, ScoreCard.js, WinProbChart.js)",
        purpose="Reusable UI elements for editing squads, showing scorecards, and charting data.",
        features=[
            "TeamBuilder.js: Form for 11 batting positions and 20 bowling overs, including auto-fill helpers.",
            "ScoreCard.js: Displays simulated scores and statistics in standard batting/bowling scorecard tables.",
            "WinProbChart.js: Renders win probability gauges and histograms of score distributions using Recharts."
        ],
        code_snippet="export default function TeamBuilder({ label, players, setPlayers, rotation, ... })"
    ))

    # Webapp Pages
    story.append(file_section(
        file_path="webapp/src/pages/ (SimulatePage.js, MonteCarloPage.js, OptimizerPage.js)",
        purpose="Connects inputs, loading states, and API requests to display results on each route page.",
        features=[
            "SimulatePage.js: Coordinates user setups to trigger a single match simulation and displays the scorecard.",
            "MonteCarloPage.js: Runs multi-match simulations, showing win probabilities, score summaries, and player projections.",
            "OptimizerPage.js: Renders forms to adjust budgets, input player registries, and display optimized batting orders and bowling rotations."
        ],
        code_snippet="export default function MonteCarloPage() {\n  // Manages inputs, triggers runMonteCarlo(), and renders visual projections\n}"
    ))

    # =========================================================================
    # SECTION 8: SYSTEM REQUIREMENTS & OPERATION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("8. Deployment &amp; Operational Checklist", h1_style))
    story.append(Paragraph(
        "To run the system locally, follow this workflow to ensure clean data preparation, "
        "model training, backend initialization, and frontend dashboard launch:",
        body_style
    ))
    
    story.append(Paragraph("<b>1. Data Alignment &amp; Prep:</b> Place your historical IPL ball-by-ball records inside `data/raw/ipl_final.csv` and run standard preprocessing:", bullet_style))
    story.append(code_box("python scripts/prepare_data.py"))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>2. Model Training (Colab):</b> Upload the generated `data/processed/features.csv` file into Google Colab. Run `notebooks/colab_training.py` (or the `.ipynb` notebook file) to download the three serialized outputs. Save them inside the local `models/` directory.", bullet_style))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>3. Backend Server:</b> Launch the FastAPI web application using uvicorn:", bullet_style))
    story.append(code_box("uvicorn src.api.app:app --reload --port 8000"))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>4. Frontend Dashboard:</b> From the frontend subdirectory, run npm dependencies installation and trigger the local developer launch:", bullet_style))
    story.append(code_box("cd webapp\nnpm install\nnpm start"))
    story.append(Spacer(1, 20))
    
    # Conclusion message
    conclusion_text = (
        "<b>Summary Note:</b> This system bridges the gap between historical player data and predictive cricket "
        "tactics. By structuring the codebase into distinct cleaning, inference, simulation, optimization, "
        "and frontend visualization layers, it enables developers and coaches to run rapid tactical "
        "evaluations based on machine-learned player outcome probabilities."
    )
    
    conclusion_box = Table([[Paragraph(conclusion_text, body_style)]], colWidths=[504])
    conclusion_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#eff6ff")),
        ('BOX', (0,0), (-1,-1), 1, HexColor("#bfdbfe")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(conclusion_box)

    # Build the document using the NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Documentation PDF created successfully: {pdf_filename}")

if __name__ == '__main__':
    main()
