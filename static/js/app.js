/* =========================================================
   DUCKPREDICT JAVASCRIPT
   ========================================================= */


// =========================================================
// PASSWORD
// =========================================================

function togglePassword() {

    const password =
        document.getElementById(
            "password"
        );

    if (!password) {
        return;
    }

    if (
        password.type ===
        "password"
    ) {

        password.type =
            "text";

    } else {

        password.type =
            "password";

    }

}


// =========================================================
// TABLE SEARCH
// =========================================================

function searchTable() {

    const input =
        document.getElementById(
            "tableSearch"
        );

    const table =
        document.getElementById(
            "datasetTable"
        );

    if (
        !input ||
        !table
    ) {
        return;
    }


    const filter =
        input.value
            .toLowerCase();


    const rows =
        table
            .getElementsByTagName(
                "tbody"
            )[0]
            .getElementsByTagName(
                "tr"
            );


    for (
        let i = 0;
        i < rows.length;
        i++
    ) {

        const text =
            rows[i]
                .textContent
                .toLowerCase();


        rows[i].style.display =
            text.includes(filter)
            ? ""
            : "none";

    }

}


// =========================================================
// PAGE ANIMATION
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const elements =
            document.querySelectorAll(
                ".panel, .stat-card, .quick-card"
            );


        elements.forEach(
            function (
                element,
                index
            ) {

                element.style.opacity =
                    "0";

                element.style.transform =
                    "translateY(8px)";


                setTimeout(
                    function () {

                        element.style.transition =
                            "opacity .4s ease, transform .4s ease";

                        element.style.opacity =
                            "1";

                        element.style.transform =
                            "translateY(0)";

                    },
                    Math.min(
                        index * 45,
                        400
                    )
                );

            }
        );

    }
);