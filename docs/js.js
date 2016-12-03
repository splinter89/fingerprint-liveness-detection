/*global LIST */

function shuffle(array) {
    var i = 0,
        j = 0,
        temp = null;

    for (i = array.length - 1; i > 0; i -= 1) {
        j = Math.floor(Math.random() * (i + 1));
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

var quiz = (function () {
    var $breadcrumbs,
        $step_title,
        $step_train,
        $step_test,
        $step_results,
        train_counter = 0,
        test_counter = 0,
        TRAIN_STEPS = 5,
        TEST_STEPS = 10,
        score = 0,
        img_list,
        test_imgs = [],
        test_is_fake = [],
        test_answers = [];

    function init() {
        $breadcrumbs = $('.breadcrumbs:first');
        $step_title = $('.step_title:first');
        $step_train = $('.step_train:first');
        $step_test = $('.step_test:first');
        $step_results = $('.step_results:first');

        shuffle(LIST.test.fake);
        shuffle(LIST.test.live);
        shuffle(LIST.train.fake);
        shuffle(LIST.train.live);

        go_to_train();
    }

    function go_to_train() {
        var $img_1,
            $img_2,
            first_is_fake = Math.random() < 0.5,
            class_1 = first_is_fake ? 'fake' : 'live',
            class_2 = first_is_fake ? 'live' : 'fake',
            src_1 = LIST.train[class_1][train_counter % LIST.train[class_1].length],
            src_2 = LIST.train[class_2][train_counter % LIST.train[class_2].length],
            $button = $step_train.find('button:first');

        $step_train.show();
        $breadcrumbs.html('<b>Training</b> &gt; Testing &gt; Results');
        $step_title.text('Question ' + (train_counter + 1) + '/' + TRAIN_STEPS);

        $img_1 = $('<img>').attr('src', src_1).addClass(class_1);
        $img_2 = $('<img>').attr('src', src_2).addClass(class_2);

        $step_train.find('.img').empty();
        $img_1.add($img_2).appendTo($step_train.find('.img')).click(function () {
            var is_fake = $(this).hasClass('fake');
            if ($button.is(':visible')) return; // already answered

            $img_1.add($img_2).css('cursor', 'default');
            $step_train.find('.img')
                .append(
                    $('<div></div>').append('<span style="margin-left:105px;">' + (first_is_fake ? 'FAKE' : 'LIVE') + '</span>')
                        .append('<span style="margin-left:235px;">' + (first_is_fake ? 'LIVE' : 'FAKE') + '</span>')
                ).append(
                    $('<div style="font-weight:bold; font-size:125%; margin-left:200px; padding:15px;"></div>')
                    .text(is_fake ? 'CORRECT' : 'WRONG')
                ).css('color', is_fake ? '#016c03' : '#ca220b');
            $button.show();
        });
    }

    function next_train(button) {
        $(button).hide();
        train_counter += 1;
        if (train_counter < TRAIN_STEPS) {
            go_to_train();
        } else {
            go_to_test();
        }
    }

    function go_to_test() {
        var is_fake = Math.random() < 0.5,
            class_1 = is_fake ? 'fake' : 'live',
            src_1 = LIST.test[class_1][test_counter % LIST.test[class_1].length];
        test_imgs.push(src_1);
        test_is_fake.push(is_fake);

        $step_train.hide();
        $step_test.show();
        $breadcrumbs.html('Training &gt; <b>Testing</b> &gt; Results');
        $step_title.text('Question ' + (test_counter + 1) + '/' + TEST_STEPS);

        $step_test.find('.img').empty();
        $('<img>').attr('src', src_1).addClass(class_1).appendTo($step_test.find('.img'));
    }

    function test_answer(is_fake) {
        var correct_is_fake = $($step_test.find('img')).hasClass('fake');
        test_answers.push(is_fake);
        if (is_fake == correct_is_fake) {
            score += 1;
        }
        next_test();
    }

    function next_test() {
        test_counter += 1;
        if (test_counter < TEST_STEPS) {
            go_to_test();
        } else {
            go_to_results();
        }
    }

    function go_to_results() {
        var i, $tr, $td;

        $step_test.hide();
        $step_results.show();
        $breadcrumbs.html('Training &gt; Testing &gt; <b>Results</b>');
        $step_title.css('visibility', 'hidden');

        img_score = Math.floor(10 * score / TEST_STEPS);
        $step_results.find('div').text('Your score: ' + score + '/' + TEST_STEPS)
            .after('<img src="img/smileys/' + img_score + '.png" alt="">');

        for (var i = 0; i < TEST_STEPS; i += 1) {
            $tr = $('<tr>').addClass((test_is_fake[i] == test_answers[i]) ? 'correct' : 'wrong');
            $td = $('<td>').append($('<img>').attr('src', test_imgs[i]));
            $tr.append($td);
            $td = $('<td>').append(test_is_fake[i] ? 'FAKE' : 'LIVE');
            $tr.append($td);
            $step_results.find('table').append($tr);
        }
    }

    return {
        init: init,
        next_train: next_train,
        test_answer: test_answer
    };
})();

$(document).ready(quiz.init);
