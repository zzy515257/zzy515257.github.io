/*
 author: @jiangwen5945 & EvanNotFound
 enhanced with dynamic date & weather by Antigravity
*/

const instances = new Map();
const initTokens = new Map();

const WMO_CODE_MAP = {
  0: "晴",
  1: "晴间多云",
  2: "多云",
  3: "阴",
  45: "大雾",
  48: "浓雾",
  51: "毛毛雨",
  53: "小雨",
  55: "小雨",
  56: "冻雨",
  57: "冻雨",
  61: "小雨",
  63: "中雨",
  65: "大雨",
  66: "冻雨",
  67: "冻雨",
  71: "小雪",
  73: "中雪",
  75: "大雪",
  77: "雪粒",
  80: "阵雨",
  81: "阵雨",
  82: "暴雨",
  85: "阵雪",
  86: "大阵雪",
  95: "雷阵雨",
  96: "雷阵雨伴有冰雹",
  99: "暴雨伴有冰雹",
};

const WEATHER_CACHE_KEY = "ZOE_BLOG_WEATHER_DALIAN";

const getCachedWeather = () => {
  try {
    const raw = sessionStorage.getItem(WEATHER_CACHE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (Date.now() - data.time < 30 * 60 * 1000) {
        return data.weatherText;
      }
    }
  } catch (e) {}
  return null;
};

const setCachedWeather = (weatherText) => {
  try {
    sessionStorage.setItem(
      WEATHER_CACHE_KEY,
      JSON.stringify({ time: Date.now(), weatherText })
    );
  } catch (e) {}
};

const getTodayDateString = () => {
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;
  const date = now.getDate();
  const weekdays = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"];
  const day = weekdays[now.getDay()];
  return `${year}年${month}月${date}日 ${day}`;
};

const normalizeSubtitleText = (subtitleText) => {
  if (Array.isArray(subtitleText)) {
    return subtitleText.filter((entry) => typeof entry === "string" && entry);
  }

  if (typeof subtitleText === "string" && subtitleText) {
    return [subtitleText];
  }

  return [];
};

const destroyInstance = (id) => {
  const instance = instances.get(id);
  if (instance && typeof instance.destroy === "function") {
    try {
      instance.destroy();
    } catch (error) {
      console.error("Failed to destroy Typed instance:", error);
    }
  }

  instances.delete(id);

  const element = document.getElementById(id);
  if (element) {
    element.innerHTML = "";
  }
};

const createTyped = (id, strings, options, retryCount = 0) => {
  if (typeof window.Typed === "undefined") {
    if (retryCount < 20) {
      setTimeout(() => createTyped(id, strings, options, retryCount + 1), 50);
    }
    return;
  }

  if (!document.getElementById(id)) {
    return;
  }

  destroyInstance(id);

  const instance = new window.Typed(`#${id}`, {
    strings,
    typeSpeed: options.typeSpeed,
    smartBackspace: options.smartBackspace,
    backSpeed: options.backSpeed,
    backDelay: options.backDelay,
    loop: options.loop,
    startDelay: options.startDelay,
  });

  instances.set(id, instance);
};

const subtitleConfig = theme?.home_banner?.subtitle || {};
const hitokotoConfig = subtitleConfig.hitokoto || {};

export const config = {
  usrTypeSpeed: subtitleConfig.typing_speed,
  usrBackSpeed: subtitleConfig.backing_speed,
  usrBackDelay: subtitleConfig.backing_delay,
  usrStartDelay: subtitleConfig.starting_delay,
  usrLoop: subtitleConfig.loop,
  usrSmartBackspace: subtitleConfig.smart_backspace,
  usrHitokotoAPI: hitokotoConfig.api,
};

export default function initTyped(id) {
  const currentToken = (initTokens.get(id) || 0) + 1;
  initTokens.set(id, currentToken);

  const {
    usrTypeSpeed,
    usrBackSpeed,
    usrBackDelay,
    usrStartDelay,
    usrLoop,
    usrSmartBackspace,
    usrHitokotoAPI,
  } = config;

  const options = {
    typeSpeed: usrTypeSpeed ?? 100,
    smartBackspace: usrSmartBackspace ?? false,
    backSpeed: usrBackSpeed ?? 60,
    backDelay: usrBackDelay ?? 2000,
    loop: usrLoop ?? true,
    startDelay: usrStartDelay ?? 500,
  };

  const hitokotoEnabled = Boolean(hitokotoConfig.enable);

  if (hitokotoEnabled) {
    if (!usrHitokotoAPI) {
      return;
    }

    fetch(usrHitokotoAPI)
      .then((response) => response.json())
      .then((data) => {
        if (initTokens.get(id) !== currentToken) {
          return;
        }

        const quote = typeof data?.hitokoto === "string" ? data.hitokoto : "";
        if (!quote) {
          return;
        }

        const author =
          typeof data?.from_who === "string" && hitokotoConfig.show_author
            ? data.from_who
            : "";
        const text = author ? `${quote}——${author}` : quote;

        createTyped(id, [text], options);
      })
      .catch((error) => {
        console.error("Failed to fetch hitokoto:", error);
      });

    return;
  }

  let subtitleEntries = normalizeSubtitleText(subtitleConfig.text);
  if (subtitleEntries.length === 0) {
    subtitleEntries = ["路虽弥，不行不至。事虽小，不做不成。"];
  }

  const dateStr = getTodayDateString();
  const cachedWeather = getCachedWeather();
  const initialWeather = cachedWeather || "晴 6℃~16℃";
  const initialDateWeather = `${dateStr} ｜ 大连 · ${initialWeather}`;

  const finalStrings = [
    subtitleEntries[0] || "路虽弥，不行不至。事虽小，不做不成。",
    initialDateWeather,
  ];

  createTyped(id, finalStrings, options);

  // Fetch live weather in the background
  fetch(
    "https://api.open-meteo.com/v1/forecast?latitude=38.914&longitude=121.6147&current_weather=true&daily=temperature_2m_max,temperature_2m_min,weathercode&timezone=Asia%2FShanghai"
  )
    .then((res) => res.json())
    .then((data) => {
      if (initTokens.get(id) !== currentToken) {
        return;
      }
      const weatherCode = data?.current_weather?.weathercode ?? 0;
      const weatherName = WMO_CODE_MAP[weatherCode] || "晴";
      const minTemp = Math.round(data?.daily?.temperature_2m_min?.[0] ?? 6);
      const maxTemp = Math.round(data?.daily?.temperature_2m_max?.[0] ?? 16);
      const liveWeather = `${weatherName} ${minTemp}℃~${maxTemp}℃`;
      setCachedWeather(liveWeather);

      const liveDateWeather = `${getTodayDateString()} ｜ 大连 · ${liveWeather}`;
      const instance = instances.get(id);
      if (instance && Array.isArray(instance.strings)) {
        instance.strings = [finalStrings[0], liveDateWeather];
      }
    })
    .catch((err) => {
      console.warn("Weather fetch failed, using fallback:", err);
    });
}
