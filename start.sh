#!/bin/bash
set -e

echo "🚀 Starting AI Knowledge Base"
echo "=============================="

source venv/bin/activate

if ! docker ps | grep -q "wandai-task2"; then
    echo "Starting Docker services..."
    docker-compose up -d
    sleep 10
fi

echo "Starting backend on http://localhost:8000"
python main.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "Starting frontend on http://localhost:8501"
echo "=============================="
echo "✅ Ready!"
echo "Backend: http://localhost:8000/docs"
echo "Frontend: http://localhost:8501"
echo "=============================="

streamlit run frontend.py --server.port 8501 --server.headless true

kill $BACKEND_PID 2>/dev/null || true
