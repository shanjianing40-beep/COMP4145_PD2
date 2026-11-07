# Completed

All of this is in one local Flask file, `app.py`. The page runs in the browser. Streamlit is not used. The skin analysis calls the HKBU GenAI URL from `需求.md`.

## Skin check

- Home has an upload area and an Analyze Skin button.
- The Upload Frontal Face Photo button stays. Clicking anywhere in the upload area, or dragging a photo onto it, also uploads the photo. A file that is not an image is rejected.
- After upload, the photo scales to fit inside the box. The box ratio is 1:1.2.
- The result shows Skin Cleanliness (/10), Skin Quality (/10), a short skin description, and an estimated age.
- Section titles `1. Source Capture`, `2. Evaluation Results`, `3. Professional Skincare Report`, and `4. Skincare Tips` sit outside the white cards and use the same size. Sections 2, 3, and 4 stay on the right, in that order.
- Clear Image and Analyze Skin are the same size, side by side under the photo box, and outside that box. The guidelines sit under those buttons, also outside the box.
- The slider track and thumb after the score are `#11D9A1`.
- Under Skin Cleanliness and Skin Quality, KPI Rating and Aesthetician Review are separate bullets, smaller than the section heading.
- Skincare Tips uses the same layout: a summary, then two tip groups with bullets.

## Login and accounts

- The login page asks for a username and password. The demo account is username `demo`, password `demo`.
- On the login card the order is Sign in, then **Create a new account**, then `Demo account: username demo, password demo`.
- The page shows one card at a time. Create a new account replaces the login card. Back returns to the login card and hides the create card.
- After a successful registration, the create card stays and says the account was created. The user clicks Back, then signs in. Registration does not enter the app by itself.
- A new account starts with an empty history and cannot see the demo photos. Uploads are stored for that username and are still there on the next login.

## History

- Home, History, and Recommend are tabs at the same level.
- History shows a line chart on the left: Skin Cleanliness in `#11D9A1` and Skin Quality in `#1f6f8b`.
- Clicking a point shows that time's photo and skincare tips on the right. The photo box follows the photo's own ratio.
- The right-hand record scrolls on its own when the photo and tips are taller than the panel.
- `What skin treatment program can improve...` is under the chart on the left side of History. The same button is also at the bottom, inside the Skincare Tips card on Home. Both open Recommend.
- The three demo photos are kept as 2026-01-15, 2026-03-15, and 2026-06-15.
- On the demo account, a new upload is remembered for the current login, including after a refresh. Log out removes those new records. The three demo photos stay. A registered account keeps its records after logout.

## Recommend

- Recommend follows the time selected on History. If none is selected, it uses the latest record.
- The left side is its own scrolling list. Each medical beauty project is in its own card, with the effect, the price in HK$, and a Reservation button.
- Clicking a project card shows that project's predicted chart on the right. The projects do not share one shape: some move cleanliness, some move quality, and the timing differs (early jump, a dip then a rise, a late jump, or a fade).
- The reservation form stays hidden until Reservation is clicked. It then appears under the chart, with a date, a time, and Pay.
- After Pay, the confirmation is a browser alert, for example: `Reserved Skin Booster Mesotherapy on 2026-11-07 at 10:00. Payment of HK$1980 received.` Payment is a demo confirmation only.
