# Use official Node LTS as base
FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --production && npm cache clean --force

COPY . .

RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
