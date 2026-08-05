import axios from "axios";

// package.json uses:
// "proxy": "http://localhost:8000"

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

export const predictFantasyXI = async (data) => {
  const res = await api.post("/predict-fantasy-xi", data);
  return res.data;
};

export default api;