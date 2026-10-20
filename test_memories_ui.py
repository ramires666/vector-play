from playwright.sync_api import sync_playwright, expect
from time import sleep
URL="http://127.0.0.1:58765"
with sync_playwright() as p:
    browser=p.chromium.launch(channel="chrome",headless=True)
    mobile=browser.new_page(viewport={"width":375,"height":812},has_touch=True)
    other=browser.new_page(viewport={"width":390,"height":844},has_touch=True)
    gallery=browser.new_page(viewport={"width":375,"height":812},has_touch=True)
    errors=[]
    for page in (mobile,other,gallery):
        page.on("pageerror",lambda err:errors.append(str(err)))
    mobile.goto(URL,wait_until="domcontentloaded")
    other.goto(URL,wait_until="domcontentloaded")
    gallery.goto(URL+"/memories",wait_until="domcontentloaded")
    assert mobile.evaluate("document.documentElement.scrollWidth<=innerWidth")
    assert mobile.locator("#liveMemories .live-item").count()>=1
    assert "королева" in gallery.locator("#items").inner_text()
    mobile.locator("#showExamples").uncheck()
    assert mobile.locator("#exampleBlock").is_hidden()
    mobile.locator("#showExamples").check()
    assert mobile.locator("#exampleBlock").is_visible()
    mobile.locator("[data-example=king]").click()
    mobile.locator("#calc").click()
    expect(mobile.locator("#calc")).to_be_enabled(timeout=60000)
    assert "королева" in mobile.locator("#results").inner_text()
    mobile.locator("#saveMemory").click()
    expect(mobile.locator("#saveFeedback")).to_contain_text("уже есть",timeout=20000)
    # create a new equation on mobile, verify it arrives on another browser and gallery
    mobile.locator("[data-example=russia]").click()
    mobile.locator("#calc").click()
    expect(mobile.locator("#calc")).to_be_enabled(timeout=60000)
    mobile.locator("#saveMemory").click()
    expect(mobile.locator("#saveMemory")).to_be_enabled(timeout=60000)
    expect(other.locator("#liveMemories .live-item")).to_have_count(2,timeout=12000)
    expect(gallery.locator("#items .card")).to_have_count(2,timeout=12000)
    assert "путин" in gallery.locator("#items").inner_text().lower()
    assert not errors,errors
    print("PASS: mobile, collapsible examples, save, dedup, 1s sync, gallery, other visitor",flush=True)
    browser.close()


