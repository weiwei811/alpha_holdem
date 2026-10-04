var socket;
$(document).ready(function() {
    // The http vs. https is important. Use http for localhost!
    socket = io();

    // Button was clicked
    document.getElementById("send_button").onclick = function() {
        // Get the text value
        var text = document.getElementById("textfield_input").value;

        // Update the chat window
        document.getElementById("chat").innerHTML += "You: " + text + "\n\n";

        // Emit a message to the 'send_message' socket
        socket.emit('send_message', {'text':text});

        // Set the textfield input to empty
        document.getElementById("textfield_input").value = "";
    }

    // Message recieved from server
    socket.on('message_from_server', function(data) {
        var text = data['text'];
        console.log(data);
        set_board(data["data"])
        //document.getElementById("chat").innerHTML += "Server push a new state." + text + "\n";
    });

    socket.on('debug_info', function(data) {
        var text = data['text'];
        var input = document.getElementById("chat");
        input.focus(); // that is because the suggest can be selected with mouse
        input.innerHTML += text + "\n";
        input.scrollTop = input.scrollHeight;

    });

    var config = {
        type: Phaser.AUTO,
        parent: 'game_container',
        width: 1024,
        height: 720,
        backgroundColor: '0xbababa',
        scene: {
            preload: preload,
            update: update_stuff
        }
    };

    var game = new Phaser.Game(config);
    var scene;
    var total_reward = 0;
    var total_match = 0;

    function send_action_to_server(action_id){
        socket.emit('send_message', {'text':"action_from_client","action_id":action_id});
    }

    function onAction(action_ind){
        socket.emit('send_message', {'text':"action_from_client","action_id":action_ind});
    }

    function set_board (infos)
    {
        if (!scene) {
            console.log("Waiting for Phaser scene to initialize...");
            setTimeout(function() { set_board(infos); }, 100);
            return;
        }
        if (scene.reg_images){
            for(var one_image of scene.reg_images){
                one_image.destroy();
            }
        }
        scene.reg_images = [];

        var done = infos["done"];
        var num_players = infos["num_players"] || 6;
        var human_id = infos["human_id"] !== undefined ? infos["human_id"] : 0;
        var current_player = infos["current_player"];
        var hands = infos["hands"] || [];
        var public_cards = infos["public"] || [];
        var all_chips = infos["chip"] || [];
        var stakes = infos["stakes"] || [];
        var pot = infos["pot"] || 0;
        var payoffs = infos["payoffs"] || [];
        var action_records = infos["action_recoards"] || [];
        var legal_actions = infos["legal_actions"] || [];

        if (done === true) {
            if (payoffs.length > human_id) {
                total_reward += payoffs[human_id];
            }
            total_match += 1;
        }

        // Draw poker table graphic (green felt with wooden border)
        var tableGfx = scene.add.graphics();
        // Wooden rim
        tableGfx.fillStyle(0x3e2723, 1);
        tableGfx.fillRoundedRect(50, 40, 730, 530, 160);
        // Table felt (rich green)
        tableGfx.fillStyle(0x1b5e20, 1);
        tableGfx.fillRoundedRect(65, 55, 700, 500, 145);
        // Inner felt border
        tableGfx.lineStyle(3, 0xffd54f, 0.4);
        tableGfx.strokeRoundedRect(85, 75, 660, 460, 130);
        scene.reg_images.push(tableGfx);

        // Community Cards & Pot in Center of Table
        var potText = scene.add.text(415, 235, 'TOTAL POT: $' + pot, {
            fontFamily: 'Arial, sans-serif',
            fontSize: '22px',
            fontStyle: 'bold',
            color: '#ffd700',
            stroke: '#000000',
            strokeThickness: 4
        }).setOrigin(0.5);
        scene.reg_images.push(potText);

        // Render Community Cards
        var pubStartX = 415 - ((public_cards.length - 1) * 35);
        for (var i = 0; i < public_cards.length; i++) {
            var cCard = public_cards[i];
            var pubImg = scene.add.image(pubStartX + i * 70, 305, cCard);
            pubImg.setScale(1.5);
            scene.reg_images.push(pubImg);
        }

        // Seating Layout for 6 Players (Oval distribution)
        // Seat 0: Hero (Bottom Center)
        // Seat 1: Bottom Left
        // Seat 2: Top Left
        // Seat 3: Top Center
        // Seat 4: Top Right
        // Seat 5: Bottom Right
        var seatPositions = [
            { x: 415, y: 520, label: "Hero (You)", isHero: true },
            { x: 140, y: 430, label: "AI 1", isHero: false },
            { x: 170, y: 140, label: "AI 2", isHero: false },
            { x: 415, y: 105, label: "AI 3", isHero: false },
            { x: 655, y: 140, label: "AI 4", isHero: false },
            { x: 685, y: 430, label: "AI 5", isHero: false }
        ];

        for (var p = 0; p < num_players; p++) {
            var pos = seatPositions[p];
            if (!pos) continue;

            var pHands = hands[p] || [];
            var pStake = stakes[p] !== undefined ? stakes[p] : 0;
            var pChip = all_chips[p] !== undefined ? all_chips[p] : 0;
            var isCurrent = (current_player === p);

            // Active player turn indicator badge / glow
            if (isCurrent && !done) {
                var turnGlow = scene.add.graphics();
                turnGlow.lineStyle(4, 0x00e676, 1);
                turnGlow.strokeCircle(pos.x, pos.y, 48);
                scene.reg_images.push(turnGlow);
            }

            // Player Cards
            var scale = pos.isHero ? 1.7 : 1.2;
            var offset = pos.isHero ? 32 : 22;
            for (var c = 0; c < 2; c++) {
                if (c < pHands.length) {
                    var cardName = pHands[c];
                    var cardSprite;
                    if (pos.isHero || done === true) {
                        cardSprite = scene.add.image(pos.x - offset + c * (offset * 2), pos.y, cardName);
                    } else {
                        cardSprite = scene.add.image(pos.x - offset + c * (offset * 2), pos.y, "card_back");
                    }
                    cardSprite.setScale(scale);
                    scene.reg_images.push(cardSprite);
                }
            }

            // Player Label & Chip Info
            var labelColor = pos.isHero ? '#4fc3f7' : (isCurrent ? '#69f0ae' : '#ffffff');
            var nameText = scene.add.text(pos.x, pos.y + (pos.isHero ? 46 : 34), pos.label, {
                fontFamily: 'Arial, sans-serif',
                fontSize: pos.isHero ? '15px' : '13px',
                fontStyle: 'bold',
                color: labelColor,
                backgroundColor: '#111111cc',
                padding: { x: 6, y: 2 }
            }).setOrigin(0.5);
            scene.reg_images.push(nameText);

            var infoStr = 'Stack: ' + pStake + (pChip > 0 ? ' | Bet: ' + pChip : '');
            if (done && payoffs[p] !== undefined && payoffs[p] !== 0) {
                infoStr += (payoffs[p] > 0 ? ' (+' + payoffs[p] + ')' : ' (' + payoffs[p] + ')');
            }
            var infoText = scene.add.text(pos.x, pos.y + (pos.isHero ? 67 : 50), infoStr, {
                fontFamily: 'Arial, sans-serif',
                fontSize: '11px',
                color: (done && payoffs[p] > 0) ? '#ffeb3b' : '#e0e0e0',
                backgroundColor: '#000000aa',
                padding: { x: 4, y: 2 }
            }).setOrigin(0.5);
            scene.reg_images.push(infoText);
        }

        // Match Stats Panel (Top Left)
        var statX = 15;
        var statY = 15;
        var winRate = total_match > 0 ? (total_reward / total_match).toFixed(1) : "0";
        var statsBox = scene.add.graphics();
        statsBox.fillStyle(0x000000, 0.7);
        statsBox.fillRoundedRect(statX, statY, 155, 65, 8);
        scene.reg_images.push(statsBox);

        scene.reg_images.push(scene.add.text(statX + 10, statY + 8, 'Total Win: ' + total_reward, {
            fontFamily: 'Arial, sans-serif', fontSize: '12px', color: total_reward >= 0 ? '#4caf50' : '#f44336'
        }));
        scene.reg_images.push(scene.add.text(statX + 10, statY + 26, 'Hands: ' + total_match, {
            fontFamily: 'Arial, sans-serif', fontSize: '12px', color: '#ffffff'
        }));
        scene.reg_images.push(scene.add.text(statX + 10, statY + 44, 'Win/Hand: ' + winRate, {
            fontFamily: 'Arial, sans-serif', fontSize: '12px', color: '#ffd54f'
        }));

        // Action History Sidebar (Right side: X=795..1015)
        var logBox = scene.add.graphics();
        logBox.fillStyle(0x000000, 0.75);
        logBox.fillRoundedRect(790, 15, 220, 560, 8);
        scene.reg_images.push(logBox);

        var logTitle = scene.add.text(900, 32, 'Action History', {
            fontFamily: 'Arial, sans-serif', fontSize: '14px', fontStyle: 'bold', color: '#00e5ff'
        }).setOrigin(0.5);
        scene.reg_images.push(logTitle);

        var startIdx = Math.max(0, action_records.length - 24);
        var lineY = 55;
        for (var i = startIdx; i < action_records.length; i++) {
            var item = action_records[i];
            var pName = item[2] || (item[0] === human_id ? "Hero" : "AI " + item[0]);
            var actStr = item[1];
            var itemColor = (item[0] === human_id) ? '#4fc3f7' : '#cccccc';
            if (actStr.indexOf('fold') >= 0) itemColor = '#e57373';
            else if (actStr.indexOf('raise') >= 0 || actStr.indexOf('allin') >= 0) itemColor = '#ffb74d';

            var logLine = scene.add.text(800, lineY, pName + ': ' + actStr, {
                fontFamily: 'Arial, sans-serif', fontSize: '12px', color: itemColor
            });
            scene.reg_images.push(logLine);
            lineY += 20;
        }

        // Action Buttons across bottom: X=110, 255, 400, 545, 690, 835, Y=665
        var btnY = 665;
        var btnStartX = 110;
        var btnSpacing = 145;
        for (var i = 0; i < legal_actions.length; i++) {
            let actionName = legal_actions[i][0];
            let isLegal = Boolean(legal_actions[i][1]);
            let ix = parseInt(i);
            let btnX = btnStartX + i * btnSpacing;

            if (isLegal) {
                let btn = scene.add.text(btnX, btnY, actionName, {
                    fontFamily: 'Arial, sans-serif',
                    fontSize: '14px',
                    fontStyle: 'bold',
                    color: '#ffffff',
                    backgroundColor: actionName === 'Fold' ? '#b71c1c' : (actionName === 'Next Game' ? '#00796b' : '#1565c0'),
                    padding: { x: 14, y: 10 }
                })
                .setOrigin(0.5)
                .setInteractive({ useHandCursor: true })
                .on('pointerdown', function() {
                    onAction(ix);
                })
                .on('pointerover', function() {
                    btn.setStyle({ fill: '#ffeb3b', backgroundColor: '#333333' });
                })
                .on('pointerout', function() {
                    btn.setStyle({
                        fill: '#ffffff',
                        backgroundColor: actionName === 'Fold' ? '#b71c1c' : (actionName === 'Next Game' ? '#00796b' : '#1565c0')
                    });
                });
                scene.reg_images.push(btn);
            } else {
                let btn = scene.add.text(btnX, btnY, actionName, {
                    fontFamily: 'Arial, sans-serif',
                    fontSize: '14px',
                    color: '#757575',
                    backgroundColor: '#263238',
                    padding: { x: 14, y: 10 }
                }).setOrigin(0.5);
                scene.reg_images.push(btn);
            }
        }
    }

    var names = [
    ]
    for(let i of "23456789TJQKA"){
        for(let j of "CDHS"){
            names.push(j + i)
        }
    }

    var game_object_names = [
    ]

    function preload ()
    {
        scene = this;
        for (var i = 0; i < names.length; i++) {
            var one_name = names[i];
            this.load.image(one_name, "static/cards/" + one_name + ".png");
        }
        this.load.image("card_back", "static/cards/card_back.png");
    }

    function update_stuff (infos)
    {
    }
    socket.emit('send_message', {'text':"start"});

});
