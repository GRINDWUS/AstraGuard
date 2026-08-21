#!/bin/bash

echo "🚀 Installing AstraGuard 3.0 Platform..."

# Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Frontend dependencies
if [ -d "dashboard" ]; then
    echo "Installing Dashboard Node packages..."
    cd dashboard
    npm install
    cd ..
elif [ -d "frontend" ]; then
    echo "Installing Frontend Node packages..."
    cd frontend
    npm install
    cd ..
fi

# Generate / Validate qualification dataset
echo "Validating qualification dataset..."
python dataset_generator/main_pipeline.py

echo ""
echo "========================================================="
echo "✅ AstraGuard 3.0 Platform Installation Complete!"
echo "========================================================="
echo ""
echo "To run the AstraGuard 3.0 platform:"
echo "  1. Terminal 1: python server.py"
echo "  2. Terminal 2: cd dashboard && npm run dev"
echo "  3. Open Browser: http://localhost:3000"
