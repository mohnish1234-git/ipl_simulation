import axios from "axios";

// The baseURL is empty because package.json specifies "proxy": "http://localhost:8000"
// Axios requests will automatically fall back to the proxy during development.
const api = axios.create({
  baseURL: "",
});

export const getMeta = async () => {
  const res = await api.get("/meta");
  return res.data;
};

export const simulateMatch = async (data) => {
  const res = await api.post("/simulate", data);
  return res.data;
};

export const runMonteCarlo = async (data) => {
  const res = await api.post("/monte-carlo", data);
  return res.data;
};

export const optimizeBat = async (data) => {
  const res = await api.post("/optimize/batting-order", data);
  return res.data;
};

export const optimizeBowl = async (data) => {
  const res = await api.post("/optimize/bowling-rotation", data);
  return res.data;
};

export const optimizeDream11 = async (data) => {
  const res = await api.post("/optimize/dream11", data);
  return res.data;
};
