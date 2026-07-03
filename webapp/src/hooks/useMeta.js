import { useState, useEffect } from "react";
import { getMeta } from "../utils/api";

export default function useMeta() {
  const [meta, setMeta]     = useState({ teams: [], venues: [], batters: [], bowlers: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMeta()
      .then(setMeta)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { meta, loading };
}
