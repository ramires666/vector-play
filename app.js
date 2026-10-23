"use strict";
/* Shared VectorPlay UI: i18n, model registry, language/model switcher. */

const VP_STRINGS = {
  en: {
    docTitle: "Vector Lab — word algebra",
    headline: "Words as vectors.",
    intro: "Add and subtract words. See which concept comes closest to the result.",
    expression: "Your expression",
    addWord: "＋ Add word",
    hideSources: "Hide source words",
    answers: "Answers",
    calculate: "Calculate →",
    calculating: "Calculating…",
    saveMemory: "⭐ Save to memories",
    allMemories: "📚 All memories",
    freshMemories: "🔥 Fresh memories",
    allLink: "All →",
    loadingCollection: "Loading the shared collection…",
    showExamples: "Show ready examples",
    examplesButton: "Examples",
    examplesHeading: "Ready examples",
    wordPlaceholder: "Type a word",
    result: "Result",
    emptyResults: "Press “Calculate” to see the nearest words.",
    footnote: "The comparison runs over the whole model vocabulary, with no hand-picked candidate list. Similarity is a mathematical score, not a probability. Subtracting a word does not remove its real influence from a concept.",
    nearest: "Nearest word",
    similarity: "Similarity",
    recognized: "Recognized",
    words: "words",
    noMatches: "No matches found",
    maxWords: "No more than 12 words",
    calcError: "Could not calculate",
    saveError: "Saving failed",
    added: "Added to the shared collection!",
    pendingNotice: "📝 Sent for review",
    already: "This example is already in the collection",
    collectionEmpty: "Nothing here yet. Add the first example!",
    repeat: "Tap to try it yourself",
    noConnection: "No connection to the collection",
    checkingModel: "Checking model",
    modelLoading: "Loading model…",
    modelReady: "Model ready",
    modelMissing: "Model not found",
    modelDownloading: "Model is downloading…",
    modelError: "Model failed to load",
    modelWaiting: "Not enough memory for two models",
    notDownloaded: "not downloaded",
    ramHint: "~{mb} MB RAM",
    serverDown: "Server unavailable",
    ariaSign: "Change sign for word {n}",
    ariaWord: "Word {n}",
    ariaDelete: "Delete word {n}",
    galleryTitle: "🔥 VECTORPLAY / MEMORIES",
    docTitleGallery: "Memories — VectorPlay",
    modeButton: "Mode",
    modeShortKids: "Kids",
    modeShortTeen: "Teens",
    modeShortOff: "All",
    modeTooltip: "Collection view mode.\n\n• For kids — only examples without moderation notes\n• For teens — everything except coarse insults\n• Show all — no filtering by notes\n\nThe strictness of checks when saving is set on the server (VECTOR_CALC_MODERATION_PROFILE).",
    modeKids: "For kids",
    modeKidsHint: "Hides examples with moderation notes: obscenities, sensitive topics",
    modeTeen: "For teens",
    modeTeenHint: "Everything except coarse insults/hate (blocks)",
    modeOff: "Show all",
    modeOffHint: "No filtering by moderation notes at all",
    modeSettings: "Settings",
    settingsMethod: "Assessment method",
    methodVectors: "By vectors",
    methodLlm: "Via LLM",
    recheckButton: "Re-assess all records",
    recheckProgress: "Re-assessing: {done} of {total}…",
    recheckDone: "Done: {done} records assessed",
    recheckNothing: "Nothing to re-assess",
    settingsToken: "Admin token (for settings and manual changes)",
    needToken: "Admin token required — open “Settings” and paste the token",
    assessLabel: "Assessment",
    byLabel: "by",
    groupLabel: "Access group",
    notAssessed: "not assessed",
    verdictOk: "safe",
    verdictReview: "questionable",
    verdictBlock: "not allowed",
    whoLlm: "LLM",
    whoVectors: "vectors",
    whoHuman: "human",
    groupKids: "Kids",
    groupTeen: "Teens",
    groupAdult: "Adults only",
    groupChangeFailed: "Could not change the group:",
    settingsHint: "Re-assessment goes through the records one by one and records who made the assessment and why.",
    modeHint: "The gear filters what is shown in this browser. Save-time checks are configured on the server.",
    shown: "shown",
    useInCalculator: "Use in calculator",
    presetLoaded: "Example loaded from the collection — press “Calculate”",
    back: "← Calculator",
    galleryHeadline: "The wild collection",
    galleryIntro: "All saved word equations, shared by visitors of this server. New examples appear automatically.",
    updating: "Updating…",
    refresh: "↻ Refresh",
    total: "Total",
    galleryEmpty: "Nothing here yet. Save the first example in the calculator.",
    shownLast: "Showing the last 100 records.",
    allModels: "All models",
    cosine: "cosine similarity",
  },
  ru: {
    docTitle: "Vector Lab — алгебра слов",
    headline: "Слова как векторы.",
    intro: "Складывай и вычитай слова. Посмотри, какое понятие ближе всего к результату.",
    expression: "Твоё выражение",
    addWord: "＋ Добавить слово",
    hideSources: "Скрыть исходные слова",
    answers: "Ответов",
    calculate: "Вычислить →",
    calculating: "Вычисляю…",
    saveMemory: "⭐ В мемориз",
    allMemories: "📚 Вся коллекция",
    freshMemories: "🔥 Свежие мемориз",
    allLink: "Все →",
    loadingCollection: "Загружаем общую коллекцию…",
    showExamples: "Показывать готовые примеры",
    examplesButton: "Примеры",
    examplesHeading: "Готовые примеры",
    wordPlaceholder: "Введи слово",
    result: "Результат",
    emptyResults: "Нажми «Вычислить», чтобы увидеть ближайшие слова.",
    footnote: "Сравнение проводится со всем словарём модели, без ручного списка кандидатов. Сходство — математический показатель, а не вероятность. Вычитание слова не означает удаление его реального влияния из понятия.",
    nearest: "Ближайшее слово",
    similarity: "Сходство",
    recognized: "Распознано",
    words: "слов",
    noMatches: "Совпадений не найдено",
    maxWords: "Не более 12 слов",
    calcError: "Не удалось вычислить",
    saveError: "Ошибка сохранения",
    added: "Добавлено в общую коллекцию!",
    pendingNotice: "📝 Отправлено на проверку",
    already: "Этот пример уже есть в коллекции",
    collectionEmpty: "Пока пусто. Добавь первый угарный пример!",
    repeat: "Нажми, чтобы попробовать самому",
    noConnection: "Нет соединения с коллекцией",
    checkingModel: "Проверка модели",
    modelLoading: "Загружаю модель…",
    modelReady: "Модель готова",
    modelMissing: "Модель не найдена",
    modelDownloading: "Модель скачивается…",
    modelError: "Модель не загрузилась",
    modelWaiting: "Не хватает памяти на две модели",
    notDownloaded: "не скачана",
    ramHint: "~{mb} МБ RAM",
    serverDown: "Сервер недоступен",
    ariaSign: "Изменить знак для слова {n}",
    ariaWord: "Слово {n}",
    ariaDelete: "Удалить слово {n}",
    galleryTitle: "🔥 VECTORPLAY / МЕМОРИЗ",
    docTitleGallery: "Мемориз — VectorPlay",
    modeButton: "Режим",
    modeShortKids: "Дети",
    modeShortTeen: "Подростки",
    modeShortOff: "Всё",
    modeTooltip: "Режим показа коллекции.\n\n• Для детей — только примеры без замечаний модерации\n• Для подростков — всё, кроме грубых оскорблений\n• Показывать всё — без фильтра по замечаниям\n\nСтрогость проверки при сохранении задаётся на сервере (VECTOR_CALC_MODERATION_PROFILE).",
    modeKids: "Для детей",
    modeKidsHint: "Скрывает примеры с замечаниями модерации: мат, спорные темы",
    modeTeen: "Для подростков",
    modeTeenHint: "Всё, кроме грубых оскорблений и травли (заблокированное скрыто)",
    modeOff: "Показывать всё",
    modeOffHint: "Совсем без фильтра по замечаниям модерации",
    modeSettings: "Настройка",
    settingsMethod: "Способ оценки",
    methodVectors: "По векторам",
    methodLlm: "Через LLM",
    recheckButton: "Перепроверить оценку вариантов",
    recheckProgress: "Переоцениваю: {done} из {total}…",
    recheckDone: "Готово: оценено {done}",
    recheckNothing: "Нечего переоценивать",
    settingsToken: "Токен админа (для настроек и ручных правок)",
    needToken: "Нужен токен админа — открой «Настройка» и вставь токен",
    assessLabel: "Оценка",
    byLabel: "кто",
    groupLabel: "Группа допуска",
    notAssessed: "без оценки",
    verdictOk: "безопасно",
    verdictReview: "спорно",
    verdictBlock: "недопустимо",
    whoLlm: "LLM",
    whoVectors: "вектора",
    whoHuman: "вручную",
    groupKids: "Дети",
    groupTeen: "Подростки",
    groupAdult: "Только взрослым",
    groupChangeFailed: "Не удалось сменить группу:",
    settingsHint: "Переоценка идёт по записям по очереди и сохраняет, кто её сделал и почему.",
    modeHint: "Шестерёнка фильтрует показ в этом браузере. Строгость проверки при сохранении настраивается на сервере.",
    shown: "показано",
    useInCalculator: "Подставить в калькулятор",
    presetLoaded: "Пример из коллекции подставлен — нажми «Вычислить»",
    back: "← Калькулятор",
    galleryHeadline: "Угарная коллекция",
    galleryIntro: "Все сохранённые словесные уравнения, общие для посетителей сервера. Новые примеры появляются автоматически.",
    updating: "Обновление…",
    refresh: "↻ Обновить",
    total: "Всего",
    galleryEmpty: "Пока здесь пусто. Сохрани первый пример в калькуляторе.",
    shownLast: "Показаны последние 100 записей.",
    allModels: "Все модели",
    cosine: "косинусное сходство",
  },
};

const VP_PRESETS = {
  en: [
    { key: "king", label: "King − man + woman", terms: [["King", 1], ["man", -1], ["woman", 1]] },
    { key: "capital", label: "Paris − France + Italy", terms: [["Paris", 1], ["France", -1], ["Italy", 1]] },
    { key: "food", label: "Sushi − Japan + Italy", terms: [["sushi", 1], ["Japan", -1], ["Italy", 1]] },
  ],
  ru: [
    { key: "king", label: "Король − мужчина + женщина", terms: [["король", 1], ["мужчина", -1], ["женщина", 1]] },
    { key: "russia", label: "Россия − Путин", terms: [["Россия", 1], ["Путин", -1]] },
    { key: "capital", label: "Россия − Москва + Франция", terms: [["Россия", 1], ["Москва", -1], ["Франция", 1]] },
  ],
};

const VP = {
  lang: "en",
  model: null,
  models: [],
  defaultModel: "en-google",
  _listeners: [],
  _langHost: null,

  t(key) {
    const table = VP_STRINGS[this.lang] || VP_STRINGS.en;
    return table[key] !== undefined ? table[key] : (VP_STRINGS.en[key] !== undefined ? VP_STRINGS.en[key] : key);
  },

  tf(key, params) {
    let text = this.t(key);
    for (const [name, value] of Object.entries(params || {})) text = text.replace("{" + name + "}", String(value));
    return text;
  },

  locale() {
    return this.lang === "ru" ? "ru-RU" : "en-US";
  },

  presets() {
    return VP_PRESETS[this.lang] || VP_PRESETS.en;
  },

  info(id) {
    return this.models.find((item) => item.id === id) || null;
  },

  onChange(listener) {
    this._listeners.push(listener);
  },

  _emit() {
    for (const listener of this._listeners) listener(this.lang, this.model);
  },

  applyStatic() {
    document.documentElement.lang = this.lang;
    document.title = this.t("docTitle");
    for (const node of document.querySelectorAll("[data-i18n]")) {
      node.textContent = this.t(node.dataset.i18n);
    }
  },

  availableLanguages() {
    return [...new Set(this.models.map((item) => item.language))];
  },

  memoriesMode() {
    const value = localStorage.getItem("vp.memoriesMode");
    return ["kids", "teen", "off"].includes(value) ? value : "teen";
  },

  setMemoriesMode(mode) {
    if (!["kids", "teen", "off"].includes(mode)) return;
    localStorage.setItem("vp.memoriesMode", mode);
    this._emit();
  },

  renderLangSwitch(host) {
    this._langHost = host || this._langHost;
    if (!this._langHost) return;
    this._langHost.replaceChildren();
    const languages = this.availableLanguages();
    // Один язык доступен (например, английские модели выключены на сервере) —
    // переключать нечего, кнопки не показываем.
    this._langHost.hidden = languages.length < 2;
    if (languages.length < 2) return;
    for (const lang of ["en", "ru"]) {
      if (!languages.includes(lang)) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "lang-button" + (lang === this.lang ? " active" : "");
      button.dataset.lang = lang;
      button.textContent = lang === "en" ? "🇬🇧 EN" : "🇷🇺 RU";
      button.setAttribute("aria-pressed", String(lang === this.lang));
      button.onclick = () => this.setLang(lang);
      this._langHost.append(button);
    }
  },

  renderModelSelect(host, { filter = false } = {}) {
    host.replaceChildren();
    if (filter) {
      const option = document.createElement("option");
      option.value = "all";
      option.textContent = this.t("allModels");
      host.append(option);
    }
    for (const info of this.models) {
      const option = document.createElement("option");
      option.value = info.id;
      option.textContent = `${info.flag} ${info.label} — ${info.name}` +
        (info.available ? ` · ${this.tf("ramHint", { mb: info.ram_mb })}` : ` (${this.t("notDownloaded")})`);
      option.disabled = !filter && !info.available;
      host.append(option);
    }
  },

  renderSwitcher(langHost, modelHost) {
    this._langHost = langHost || this._langHost;
    this._modelHost = modelHost || this._modelHost;
    this.renderLangSwitch(this._langHost);
    this.renderModelSelect(this._modelHost);
    this._modelHost.value = this.model || this.defaultModel;
    this._bindModelSelect();
  },

  _bindModelSelect() {
    if (!this._modelHost) return;
    this._modelHost.onchange = () => this.selectModel(this._modelHost.value);
  },

  async refreshModels() {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) throw new Error("models request failed");
    const data = await response.json();
    this.models = data.models || [];
    this.defaultModel = data.default || this.defaultModel;
    return data;
  },

  defaultFor(lang) {
    const stored = localStorage.getItem(`vp.lastModel.${lang}`);
    if (stored && this.info(stored)) return stored;
    const match = this.models.find((item) => item.language === lang);
    return match ? match.id : this.defaultModel;
  },

  async setLang(lang) {
    if (!this.availableLanguages().includes(lang)) return;
    await this.selectModel(this.defaultFor(lang));
  },

  async selectModel(id) {
    const info = this.info(id);
    if (!info) return;
    this.model = id;
    this.lang = info.language;
    localStorage.setItem("vp.model", id);
    localStorage.setItem("vp.lang", this.lang);
    localStorage.setItem(`vp.lastModel.${this.lang}`, id);
    this.applyStatic();
    if (this._langHost) this.renderLangSwitch(this._langHost);
    if (this._modelHost) {
      this.renderModelSelect(this._modelHost);
      this._modelHost.value = id;
      this._bindModelSelect();
    }
    this._emit();
    try {
      await fetch(`/api/models/${encodeURIComponent(id)}/load`, { method: "POST" });
    } catch (error) {
      /* статус подтянет опрос /api/status */
    }
  },

  async init(langHost, modelHost) {
    this._langHost = langHost;
    this._modelHost = modelHost;
    const storedLang = localStorage.getItem("vp.lang");
    const storedModel = localStorage.getItem("vp.model");
    this.lang = storedLang === "ru" ? "ru" : "en";
    try {
      await this.refreshModels();
    } catch (error) {
      this.models = [];
    }
    const languages = this.availableLanguages();
    const storedAvailable = storedModel && this.info(storedModel) ? storedModel : null;
    // Если сохранённая модель выключена на сервере (например, английские забыты
    // в localStorage, а на сервере осталась только русская) — берём язык из того,
    // что реально включено, и не мигаем недоступным.
    if (!storedAvailable && languages.length === 1) this.lang = languages[0];
    const model = storedAvailable || this.defaultFor(this.lang);
    if (model) {
      await this.selectModel(model);
    } else {
      this.applyStatic();
      this.renderSwitcher(this._langHost, this._modelHost);
      this._emit();
    }
  },

  async initLangOnly(langHost) {
    this._langHost = langHost;
    const storedLang = localStorage.getItem("vp.lang");
    this.lang = storedLang === "ru" ? "ru" : "en";
    try {
      await this.refreshModels();
    } catch (error) {
      this.models = [];
    }
    // Если доступен один язык (например, английские модели выключены) — сразу его.
    const languages = this.availableLanguages();
    if (!storedLang && languages.length === 1) this.lang = languages[0];
    this.applyStatic();
    this.renderLangSwitch(langHost);
    this._emit();
  },
};
