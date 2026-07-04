import { useState, useEffect } from "react";
import { getMeta } from "../utils/api";

const EMPTY_META = { teams: [], venues: [], batters: [], bowlers: [] };

export default function useMeta() {
  const [meta, setMeta]     = useState(EMPTY_META);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    const loadMeta = async () => {
      try {
        const apiMeta = await getMeta();
        if (!cancelled) setMeta({ ...EMPTY_META, ...apiMeta });
      } catch (apiError) {
        try {
          const fallbackRes = await fetch("/meta.json");
          if (!fallbackRes.ok) throw new Error("Local metadata unavailable");
          const fallbackMeta = await fallbackRes.json();
          if (!cancelled) setMeta({ ...EMPTY_META, ...fallbackMeta });
        } catch (fallbackError) {
          if (!cancelled) {
            setMeta(EMPTY_META);
            setError(apiError?.message || fallbackError?.message || "Unable to load metadata");
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    loadMeta();
    return () => {
      cancelled = true;
    };
  }, []);

  return { meta, loading, error };
}
