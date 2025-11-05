from flask import Flask, request, jsonify, render_template
from functools import wraps
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

# Ensure required directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Logging Configuration
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Flask App Initialization
app = Flask(__name__)

# Configuration
CSV_FILE = os.path.join('data', 'attendance.csv')
TIFFIN_PRICES = {
    'half': 60,
    'full': 80
}

def log_operation(func):
    """A decorator for logging entry, exit, and errors in functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"START - {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"END - {func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.exception(f"ERROR in {func.__name__}: {str(e)}")
            raise
    return wrapper

@log_operation
def load_data():
    """Load and clean attendance data from CSV."""
    try:
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w') as f:
                f.write("date,name,type,points,total_points,price\n")
            return pd.DataFrame(columns=["date", "name", "type", "points", "total_points", "price"])
            
        df = pd.read_csv(CSV_FILE)
        logger.info(f"Raw data loaded. Columns: {df.columns.tolist()}")
        
        # Ensure all required columns exist
        required_cols = ["date", "name", "type", "points", "total_points", "price"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Convert date column to datetime, handle different date formats
        if not df.empty and 'date' in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
            # Fill missing dates with the most recent date or today's date if no dates exist
            if df['date'].isna().all() or df['date'].isnull().all():
                df['date'] = pd.Timestamp.now().normalize()
            else:
                # For rows with missing dates, use the most recent date in the file
                most_recent_date = df['date'].max()
                df['date'] = df['date'].fillna(most_recent_date)
        
        # Ensure numeric columns are properly typed
        for col in ['points', 'total_points', 'price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure type is string and lowercase
        if 'type' in df.columns:
            df['type'] = df['type'].astype(str).str.lower()
            
        # Recalculate price based on type to ensure it's correct
        df['price'] = df['type'].map(TIFFIN_PRICES).fillna(0)
        
        # Log some statistics
        logger.info(f"Loaded {len(df)} entries")
        if not df.empty:
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Latest entries:\n{df.tail(3).to_string()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=["date", "name", "type", "points", "total_points", "price"])

@log_operation
def save_data(df):
    """Save DataFrame to CSV."""
    try:
        df.to_csv(CSV_FILE, index=False)
        logger.info("Data saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        return False

# Routes
@app.route('/')
def index():
    """Serve the main index.html page."""
    return render_template('index.html')

@app.route('/api/entry', methods=['POST'])
@log_operation
def add_entry():
    """Add a new entry to the log."""
    try:
        entry = request.get_json()
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in entry or not entry[field]:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Calculate points and price based on tiffin type
        tiffin_type = entry["type"].lower()
        entry["points"] = 1 if tiffin_type == "half" else 2
        entry["total_points"] = entry["points"]
        entry["price"] = TIFFIN_PRICES.get(tiffin_type, 0)
        
        # Add current date if not provided
        if 'date' not in entry or not entry['date']:
            entry['date'] = datetime.now().strftime('%Y-%m-%d')
        
        df = load_data()
        new_row = pd.DataFrame([entry])
        df = pd.concat([df, new_row], ignore_index=True)
        
        if save_data(df):
            return jsonify({
                "status": "success", 
                "message": "Entry added successfully",
                "price": entry["price"]
            })
        else:
            return jsonify({"error": "Failed to save data"}), 500
            
    except Exception as e:
        logger.exception(f"Failed to add entry: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/summary', methods=['GET'])
@log_operation
def get_summary():
    """Get both weekly and complete price summaries."""
    try:
        df = load_data()
        
        if df.empty:
            return jsonify({
                "weekly_summary": [],
                "weekly_total": 0,
                "complete_summary": [],
                "complete_total": 0,
                "csv": "name,half_tiffins,full_tiffins,total_tiffins,total_price\nNo data available"
            })
        
        # Ensure date is datetime and filter out invalid dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].copy()  # Remove rows with invalid dates
        
        # Ensure price is numeric and fill any missing values
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        
        logger.info(f"Total valid entries in database: {len(df)}")
        
        # Count half and full tiffins
        df['half_count'] = (df['type'].str.lower() == 'half').astype(int)
        df['full_count'] = (df['type'].str.lower() == 'full').astype(int)
        
        # Get current week's data (last 7 days including today)
        today = pd.Timestamp.now().normalize()
        week_ago = (today - pd.Timedelta(days=6)).normalize()  # 7 days including today
        
        # Log the date range we're looking for
        logger.info(f"Looking for entries between {week_ago.date()} and {today.date()}")
        
        # Ensure we're comparing dates properly and handle timezone-naive/aware comparison
        week_data = df[
            df['date'].notna() & 
            (df['date'].dt.normalize() >= week_ago) & 
            (df['date'].dt.normalize() <= today)
        ].copy()
        
        logger.info(f"Found {len(week_data)} entries in this period")
        
        # Function to generate summary
        def generate_summary(data):
            if data.empty:
                return [], 0
                
            summary = data.groupby('name').agg({
                'half_count': 'sum',
                'full_count': 'sum',
                'price': 'sum'
            }).reset_index()
            
            summary['total_tiffins'] = summary['half_count'] + summary['full_count']
            summary['total_price'] = summary['price'].round(2)
            
            total = float(summary['price'].sum().round(2))
            return summary[['name', 'half_count', 'full_count', 'total_tiffins', 'total_price']].to_dict('records'), total
        
        # Generate both summaries
        weekly_summary, weekly_total = generate_summary(week_data)
        complete_summary, complete_total = generate_summary(df)
        
        # Generate CSV for complete data
        csv_data = "name,half_tiffins,full_tiffins,total_tiffins,total_price\n"
        if complete_summary:
            for row in complete_summary:
                csv_data += f"{row['name']},{row['half_count']},{row['full_count']},{row['total_tiffins']},{row['total_price']:.2f}\n"
            total_half = sum(row['half_count'] for row in complete_summary)
            total_full = sum(row['full_count'] for row in complete_summary)
            total_tiffins = sum(row['total_tiffins'] for row in complete_summary)
            csv_data += f"TOTAL,{total_half},{total_full},{total_tiffins},{complete_total:.2f}"
        
        return jsonify({
            "weekly_summary": weekly_summary,
            "weekly_total": weekly_total,
            "complete_summary": complete_summary,
            "complete_total": complete_total,
            "csv": csv_data
        })
        
    except Exception as e:
        logger.exception(f"Failed to generate price summary: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write("date,name,type,points,total_points,price\n")
    
    logger.info("Starting Tiffin Tracker server...")
    app.run(debug=True, port=5000)