const express = require('express');
const app = express();
const path = require('path');

// 加入跨域隔離所需的 HTTP 標頭
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});

// 提供 public 目錄作為靜態檔案的根目錄
app.use(express.static(path.join(__dirname, 'public')));

// 啟動伺服器，監聽 8080 port
app.listen(8080, () => {
  console.log('Server running at http://localhost:8080');
});
