#!/bin/bash

echo "🔍 Verifying AstraGuard 3.0 Setup & Health..."
echo ""

# Check Python environment
echo "✓ Python version:"
python --version

# Check Python dependencies
echo ""
echo "✓ Checking Python core dependencies..."
python -c "import pandas, numpy, scipy, xgboost, sklearn, torch, fastapi; print('  ✅ All core Python ML & Web packages verified!')"

# Check dataset presence
echo ""
echo "✓ Checking qualification dataset files..."
if [ -f "validation/dataset/test_lot_4.csv" ]; then
    lines=$(wc -l < validation/dataset/test_lot_4.csv)
    echo "  ✅ Qualification dataset found ($lines rows in test_lot_4.csv)"
elif [ -f "astraguard_core/data/LOT_2026_07.csv" ]; then
    lines=$(wc -l < astraguard_core/data/LOT_2026_07.csv)
    echo "  ✅ Qualification dataset found ($lines rows in LOT_2026_07.csv)"
else
    echo "  ❌ Dataset missing! Run python dataset_generator/main_pipeline.py"
    exit 1
fi

# Check backend server readiness
echo ""
echo "✓ Checking backend entry point..."
if [ -f "server.py" ]; then
    echo "  ✅ FastAPI backend script (server.py) verified"
else
    echo "  ❌ server.py missing!"
    exit 1
fi

echo ""
echo "========================================================="
echo "✅ ALL ASTRAGUARD 3.0 HEALTH CHECKS PASSED!"
echo "========================================================="
echo ""
echo "Ready to launch demo:"
echo "  1. python server.py"
echo "  2. cd dashboard && npm run dev"
echo "  3. Open http://localhost:3000"
