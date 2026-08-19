FROM node:22-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN npm install -g corepack@latest \
  && corepack pnpm install \
  && python3 -m pip install --break-system-packages --no-cache-dir -r requirements-python.txt \
  && corepack pnpm run build

ENV NODE_ENV=production
ENV APP_ROOT=/app
ENV PYTHON_BIN=/usr/bin/python3
CMD ["node", "dist/index.js"]
