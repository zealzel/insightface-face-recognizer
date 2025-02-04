const { legacyCreateProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  app.use(
    ["/recognize"],
    legacyCreateProxyMiddleware({
      target: "http://127.0.0.1:5000",
      changeOrigin: true,
    }),
  );
};
