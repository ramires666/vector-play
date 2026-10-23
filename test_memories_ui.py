from playwright.sync_api import sync_playwright, expect
URL="http://127.0.0.1:58765"
with sync_playwright() as p:
    browser=p.chromium.launch(channel="chrome",headless=True)
    mobile=browser.new_page(viewport={"width":375,"height":812},has_touch=True)
    other=browser.new_page(viewport={"width":390,"height":844},has_touch=True)
    gallery=browser.new_page(viewport={"width":375,"height":812},has_touch=True)
    init="localStorage.setItem('vp.lang','ru');localStorage.setItem('vp.model','ru-rusvectores');"
    for page in (mobile,other,gallery):
        page.add_init_script(init)
    errors=[]
    for page in (mobile,other,gallery):
        page.on("pageerror",lambda err:errors.append(str(err)))
    mobile.goto(URL,wait_until="domcontentloaded")
    other.goto(URL,wait_until="domcontentloaded")
    gallery.goto(URL+"/memories",wait_until="domcontentloaded")
    assert mobile.evaluate("document.documentElement.scrollWidth<=innerWidth")
    mobile.locator("#liveMemories .live-item").first.wait_for(timeout=12000)
    assert mobile.locator("#liveMemories .live-item").count()>=1
    expect(gallery.locator("#items")).to_contain_text("королева",timeout=12000)
    # «Примеры и мемориз» — один блок: пока калькулятором не пользовались,
    # сверху готовые примеры, ниже свежие из коллекции
    expect(mobile.locator("#exampleBlock")).to_be_visible(timeout=5000)
    expect(mobile.locator("#exampleBlock .preset").first).to_be_visible(timeout=5000)
    mobile.locator("[data-example=king]").click()
    mobile.locator("#calc").click(timeout=300000)
    expect(mobile.locator("#calc")).to_be_enabled(timeout=60000)
    assert "королева" in mobile.locator("#results").inner_text()
    # после первого расчёта захардкоженные примеры убираются (база не пуста)
    expect(mobile.locator("#exampleBlock")).to_be_hidden(timeout=5000)
    mobile.locator("#saveMemory").click()
    expect(mobile.locator("#saveFeedback")).to_contain_text("уже есть",timeout=20000)
    # новое уравнение набираем руками — кнопок готовых примеров уже нет
    mobile.locator(".phrase").nth(0).fill("Россия")
    mobile.locator(".phrase").nth(1).fill("Путин")
    mobile.locator(".phrase").nth(2).fill("")
    mobile.locator("#calc").click(timeout=300000)
    expect(mobile.locator("#calc")).to_be_enabled(timeout=60000)
    mobile.locator("#saveMemory").click()
    expect(mobile.locator("#saveMemory")).to_be_enabled(timeout=60000)
    expect(other.locator("#liveMemories")).to_contain_text("Путин",timeout=12000)
    expect(gallery.locator("#items")).to_contain_text("Путин",timeout=12000)
    # «убираем навсегда»: после перезагрузки готовые примеры тоже скрыты
    mobile.reload(wait_until="domcontentloaded")
    mobile.locator("#liveMemories .live-item").first.wait_for(timeout=12000)
    expect(mobile.locator("#exampleBlock")).to_be_hidden(timeout=5000)
    assert not errors,errors
    print("PASS: mobile, merged examples+memories, save, dedup, 1s sync, gallery, other visitor",flush=True)
    browser.close()

