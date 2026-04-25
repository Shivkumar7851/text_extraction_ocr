import app from "./app.js";
import router from "./routes/document.routes.js";

const PORT = 3000;

app.use('/api',router);

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});