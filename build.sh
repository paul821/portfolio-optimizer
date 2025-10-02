#!/bin/bash
# build.sh - Vercel build script for Streamlit

# Install Python dependencies
pip install -r requirements.txt

# Create a simple start script
echo "Creating start script..."
cat > start.sh << 'EOF'
#!/bin/bash
streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
EOF

chmod +x start.sh

echo "Build complete!"
